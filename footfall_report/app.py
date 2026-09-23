"""Footfall report - Streamlit front-end.

Workflow: upload CSV -> review the column mapping -> process -> report.
Run with:  streamlit run app.py
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import streamlit as st

from footfall import analysis
from footfall.config import DEFAULT_CONFIG_PATH, Config
from footfall.issues import issues_frame
from footfall.parsing import (
    COMBINED_FIELDS,
    FIELD_LABELS,
    SOURCE_ROW,
    SPLIT_FIELDS,
    detect_mode,
    parse,
    read_raw,
    resolve_mapping,
)

st.set_page_config(page_title="Footfall report", page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 2.2rem; max-width: 1500px;}
      div[data-testid="stMetric"] {
          background: rgba(128,128,128,0.07);
          border: 1px solid rgba(128,128,128,0.20);
          border-radius: 10px;
          padding: 0.75rem 1rem;
      }
      div[data-testid="stMetricValue"] {font-size: 1.6rem;}
      h1 {font-size: 2.0rem; margin-bottom: 0.2rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

def _stretch_kwargs() -> dict:
    """Streamlit renamed the 'fill the container' flag in 1.49; support both spellings."""
    try:
        major, minor = (int(part) for part in st.__version__.split(".")[:2])
    except (ValueError, AttributeError):
        return {"use_container_width": True}
    return {"width": "stretch"} if (major, minor) >= (1, 49) else {"use_container_width": True}


STRETCH = _stretch_kwargs()

DELIMITERS = {"Auto-detect": None, "Comma  ,": ",", "Semicolon  ;": ";", "Tab": "\t", "Pipe  |": "|"}
TIME_MODES = {
    "Auto-detect": "auto",
    "Separate date + hour columns": "split",
    "Single timestamp column": "combined",
}


@st.cache_data(show_spinner=False)
def load_base_config() -> Config:
    return Config.load()


def sidebar_settings(base: Config) -> tuple[Config, str]:
    st.sidebar.header("Settings")
    st.sidebar.caption(
        f"Defaults are read from `{DEFAULT_CONFIG_PATH.name}`. Changes here apply to this session only."
    )

    mode_label = st.sidebar.selectbox(
        "Time columns",
        list(TIME_MODES),
        index=list(TIME_MODES.values()).index(base.time_mode) if base.time_mode in TIME_MODES.values() else 0,
    )
    delimiter_label = st.sidebar.selectbox("Delimiter", list(DELIMITERS), index=0)
    encoding = st.sidebar.text_input("File encoding", base.encoding)
    dayfirst = st.sidebar.checkbox("Dates are day-first (DD/MM/YYYY)", base.dayfirst)

    st.sidebar.subheader("Validation")
    allow_sub_hourly = st.sidebar.checkbox(
        "Accept sub-hourly timestamps",
        base.allow_sub_hourly,
        help="Off: a 10:30 timestamp is rejected and reported. On: it is bucketed into the 10:00 hour.",
    )
    allow_negative = st.sidebar.checkbox("Accept negative counts", base.allow_negative_counts)
    floors_text = st.sidebar.text_input(
        "Expected floors (comma separated)",
        ", ".join(base.allowed_floors),
        help="Leave empty to accept any floor label. Filled in: any other label is rejected and reported.",
    )
    max_count_text = st.sidebar.text_input(
        "Max plausible count per hour",
        "" if base.max_plausible_hourly_count is None else str(base.max_plausible_hourly_count),
        help="Leave empty for no upper bound. Larger values are rejected as implausible.",
    )
    max_count = None
    if max_count_text.strip():
        try:
            max_count = int(float(max_count_text))
        except ValueError:
            st.sidebar.error("Max plausible count must be a number - ignoring it.")

    cfg = replace(
        base,
        time_mode=TIME_MODES[mode_label],
        delimiter=DELIMITERS[delimiter_label],
        encoding=encoding.strip() or "utf-8-sig",
        dayfirst=dayfirst,
        allow_sub_hourly=allow_sub_hourly,
        allow_negative_counts=allow_negative,
        allowed_floors=[f.strip() for f in floors_text.split(",") if f.strip()],
        max_plausible_hourly_count=max_count,
    )
    return cfg, mode_label


def mapping_editor(headers: list[str], cfg: Config, mode: str) -> dict[str, str | None]:
    """Show the auto-detected mapping and let the user correct it before processing."""
    detected = resolve_mapping(headers, cfg, mode)
    fields = COMBINED_FIELDS if mode == "combined" else SPLIT_FIELDS
    options = ["(not mapped)"] + headers
    mapping: dict[str, str | None] = {}

    columns = st.columns(len(fields))
    for column, field_name in zip(columns, fields):
        current = detected.get(field_name)
        index = options.index(current) if current in options else 0
        with column:
            choice = st.selectbox(
                FIELD_LABELS[field_name],
                options,
                index=index,
                key=f"map_{field_name}",
            )
        mapping[field_name] = None if choice == "(not mapped)" else choice
    return mapping


def format_table(table: pd.DataFrame) -> pd.DataFrame:
    """Display copy: percentages as text, everything else left numeric for sorting."""
    shown = table.copy()
    if "Error rate %" in shown.columns:
        shown["Error rate %"] = shown["Error rate %"].map(
            lambda v: "-" if pd.isna(v) else f"{v:.1f}%"
        )
    return shown


def show_table(table: pd.DataFrame, filename: str, height: int | None = None) -> None:
    extra = {"height": height} if height else {}
    st.dataframe(format_table(table), hide_index=True, **extra, **STRETCH)
    st.download_button(
        "Download as CSV",
        table.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        key=f"dl_{filename}",
    )


def render_kpis(kpi: dict) -> None:
    row = st.columns(4)
    row[0].metric("Total IN", f"{kpi['total_in']:,}")
    row[1].metric("Total OUT", f"{kpi['total_out']:,}")
    row[2].metric("IN - OUT", f"{kpi['difference']:+,}")
    row[3].metric(
        "Error rate",
        "-" if kpi["error_rate"] is None else f"{kpi['error_rate']:.1f}%",
        help="|IN - OUT| / (IN + OUT). The share of all events that cannot be matched to a counterpart.",
    )

    row = st.columns(4)
    row[0].metric("Floors", kpi["floors"])
    row[1].metric("Days", kpi["days"])
    row[2].metric(
        "Busiest day",
        kpi["busiest_day"] or "-",
        help=f"{kpi['busiest_day_events']:,} events (IN + OUT) across all floors.",
    )
    row[3].metric(
        "Busiest hour",
        kpi["busiest_hour"] or "-",
        help=f"{kpi['busiest_hour_events']:,} events summed over all days and floors. "
        f"Single busiest slot: {kpi['busiest_slot']}.",
    )


def render_report(result, filename: str, cfg: Config) -> None:
    clean = result.clean
    rejected = result.rejected
    issues = result.issues + analysis.coverage_issues(clean)

    valid_rows = len(clean)
    rejected_rows = len(rejected)

    if not result.ok:
        st.error(f"**{filename}** could not be turned into a report. See the findings below.")
    else:
        st.success(
            f"**{filename}** processed - {result.raw_row_count:,} data rows read, "
            f"{valid_rows:,} valid, {rejected_rows:,} rejected."
        )
    if rejected_rows:
        st.warning(
            f"{rejected_rows:,} row(s) failed validation and are **excluded from every table and "
            "total below**. They are listed unchanged in the Data quality tab."
        )

    if result.ok:
        st.subheader("Summary")
        render_kpis(analysis.kpis(clean))

    hourly = analysis.hourly_table(clean)
    by_floor = analysis.daily_by_floor(clean)
    totals = analysis.daily_totals(clean)

    error_count = sum(1 for i in issues if i.severity == "error")
    warning_count = sum(1 for i in issues if i.severity == "warning")
    quality_label = "Data quality"
    if error_count or warning_count:
        quality_label += f" ({error_count + warning_count})"

    tab_totals, tab_floor, tab_hourly, tab_quality = st.tabs(
        ["Daily totals", "Daily totals by floor", "Hourly detail", quality_label]
    )

    with tab_totals:
        st.markdown("**Daily totals across all floors**")
        show_table(totals, "daily_totals.csv")
    with tab_floor:
        st.markdown("**Daily totals by floor**")
        show_table(by_floor, "daily_totals_by_floor.csv")
    with tab_hourly:
        st.markdown(f"**Hourly IN/OUT per floor** - {len(hourly):,} rows")
        show_table(hourly, "hourly_detail.csv", height=520)

    with tab_quality:
        st.markdown("**Findings**")
        if issues:
            st.dataframe(issues_frame(issues), hide_index=True, **STRETCH)
        else:
            st.success("No data-quality issues found.")

        st.markdown("**Rejected rows** (excluded from all totals, shown exactly as in the file)")
        if rejected_rows:
            st.dataframe(rejected, hide_index=True, **STRETCH)
            st.download_button(
                "Download rejected rows",
                rejected.to_csv(index=False).encode("utf-8"),
                file_name="rejected_rows.csv",
                mime="text/csv",
                key="dl_rejected",
            )
        else:
            st.info("No rows were rejected.")

        with st.expander("How the file was interpreted"):
            st.write(
                {
                    "delimiter": repr(result.delimiter),
                    "time mode": result.mode,
                    "column mapping": {FIELD_LABELS[k]: v for k, v in result.mapping.items()},
                    "columns not used": result.extra_columns or "(none)",
                    "expected floors": cfg.allowed_floors or "(any accepted)",
                    "sub-hourly timestamps": "bucketed into the hour" if cfg.allow_sub_hourly else "rejected",
                }
            )


# --------------------------------------------------------------------------- page


def main() -> None:
    st.title("Footfall report")
    st.caption(
        "Upload the hourly IN/OUT export, check the column mapping, and generate the daily report."
    )

    config, _ = sidebar_settings(load_base_config())

    uploaded = st.file_uploader("1. Upload the CSV file", type=["csv", "txt"])
    if uploaded is None:
        st.info(
            "Waiting for a CSV file. Column names are detected automatically and can be corrected "
            "before processing; the accepted names live in `config.yaml`."
        )
        return

    file_bytes = uploaded.getvalue()
    try:
        raw_preview, meta = read_raw(file_bytes, config)
    except ValueError as exc:
        st.error(str(exc))
        return

    headers = [c for c in raw_preview.columns if c != SOURCE_ROW]
    mode = detect_mode(headers, config)

    st.markdown("### 2. Column mapping")
    st.caption(
        f"Delimiter: `{meta['delimiter']}` - {len(raw_preview):,} data rows - time format detected as: "
        f"{'single timestamp column' if mode == 'combined' else 'separate date + hour columns'}"
    )
    mapping = mapping_editor(headers, config, mode)

    with st.expander("Preview the first rows of the file"):
        st.dataframe(raw_preview.head(10), hide_index=True, **STRETCH)

    st.markdown("### 3. Process")
    if st.button("Process file", type="primary"):
        with st.spinner("Validating and aggregating..."):
            st.session_state["result"] = parse(file_bytes, config, mapping=mapping, mode=mode)
            st.session_state["filename"] = uploaded.name

    result = st.session_state.get("result")
    if result is None:
        st.info("Press **Process file** to generate the report.")
        return

    st.divider()
    render_report(result, st.session_state.get("filename", uploaded.name), config)


if __name__ == "__main__":
    main()
