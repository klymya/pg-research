"""Render the evaluation result as Markdown, then convert to PDF.

The Markdown is written into a temp directory together with PNG assets,
then `markdown` + `weasyprint` produce the final PDF. The intermediate
Markdown is kept next to the PDF only if `keep_md=True`.
"""

from __future__ import annotations

import datetime as _dt
import logging
import shutil
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from .evaluate import EvaluationResult
from .narrative import narrate
from .preprocess import MODE as PREPROCESS_MODE
from .verdict import MetricResult


log = logging.getLogger(__name__)

# R, G, B channel index + plot colour, used for the histogram panels.
_RGB_CHANNELS: tuple[tuple[str, int, str], ...] = (
    ("R", 0, "tab:red"),
    ("G", 1, "tab:green"),
    ("B", 2, "tab:blue"),
)


VERDICT_BANNERS: dict[str, tuple[str, str]] = {
    "pass": ("PASS", "DUT sharpness is same or better than reference."),
    "marginal": ("MARGINAL", "DUT is borderline on sharpness — manual review recommended."),
    "fail": ("FAIL", "DUT is softer than reference."),
}

PER_IMAGE_BANNERS: dict[str, str] = {
    "pass": "sharpness within tolerance.",
    "marginal": "one blocking metric (sharpness) outside tolerance.",
    "fail": "sharpness outside tolerance.",
}


CSS = """
@page { size: A4; margin: 18mm 16mm; }
body { font-family: "DejaVu Sans", "Helvetica", sans-serif; font-size: 10pt; color: #222; }
h1 { font-size: 18pt; margin-bottom: 0.2em; }
h2 { font-size: 13pt; margin-top: 1.4em; border-bottom: 1px solid #ccc; padding-bottom: 0.2em; }
h3 { font-size: 11pt; margin-top: 1.2em; }
table { border-collapse: collapse; width: 100%; margin: 0.5em 0 1em 0; font-size: 9pt; }
th, td { border: 1px solid #bbb; padding: 4px 8px; text-align: right; }
th { background: #eef; text-align: center; }
td:first-child, th:first-child { text-align: left; }
img { max-width: 100%; height: auto; }
.banner { padding: 10px 14px; border-radius: 4px; margin: 0.6em 0 1em 0; font-weight: bold; }
.banner.pass { background: #e6f4ea; color: #1e6e3a; border: 1px solid #1e6e3a; }
.banner.marginal { background: #fff4e5; color: #8a5a00; border: 1px solid #8a5a00; }
.banner.fail { background: #fdecea; color: #8a1f1f; border: 1px solid #8a1f1f; }
.banner.small { padding: 6px 10px; font-size: 9.5pt; margin: 0.3em 0 0.6em 0; }
.legend { font-size: 9pt; color: #444; }
.conclusion { background: #f4f6fb; border-left: 3px solid #6688cc; padding: 6px 10px; margin: 0.4em 0; font-size: 10pt; }
.disclaimer { background: #fff8e1; border-left: 3px solid #c08a00; padding: 8px 12px; margin: 0.6em 0 1em 0; font-size: 9pt; color: #5a4500; }
.ok { color: #1e6e3a; font-weight: bold; }
.bad { color: #8a1f1f; font-weight: bold; }
.warn { color: #8a5a00; font-weight: bold; }
.advisory { background: #fff4e5; border-left: 3px solid #8a5a00; padding: 8px 12px; margin: 0.6em 0 1em 0; font-size: 9.5pt; color: #5a3a00; }
.sev { font-size: 8pt; color: #555; }
"""


def _fmt(v: float, digits: int = 2) -> str:
    if v is None or (isinstance(v, float) and (np_isnan(v))):
        return "—"
    return f"{v:.{digits}f}"


def np_isnan(v) -> bool:
    try:
        return bool(np.isnan(v))
    except (TypeError, ValueError):
        return False


def _within_glyph(m: MetricResult) -> str:
    """Tolerance cell: green OK, red OUT for blocking, amber OUT for advisory
    (outside tolerance but not grounds for failing the unit).
    """
    if m.within_tol:
        return '<span class="ok">OK</span>'
    if m.is_blocking:
        return '<span class="bad">OUT</span>'
    return '<span class="warn">OUT</span>'


def _save_side_by_side(
    ref_img: np.ndarray,
    dut_img: np.ndarray,
    ref_label: str,
    dut_label: str,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(ref_img)
    axes[0].set_title(f"REF: {ref_label}", fontsize=10)
    axes[0].axis("off")
    axes[1].imshow(dut_img)
    axes[1].set_title(f"DUT: {dut_label}", fontsize=10)
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _save_pair_histogram(
    ref_img: np.ndarray,
    dut_img: np.ndarray,
    out_path: Path,
    title: str = "RGB histogram — REF (solid) vs DUT (dashed)",
) -> None:
    """3-panel R/G/B histogram, ref solid + DUT dashed in the same axes."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharey=True)
    for ax, (name, idx, color) in zip(axes, _RGB_CHANNELS):
        ax.hist(
            ref_img[..., idx].ravel(), bins=64, range=(0, 256),
            histtype="step", color=color, linewidth=1.4, label="ref",
        )
        ax.hist(
            dut_img[..., idx].ravel(), bins=64, range=(0, 256),
            histtype="step", color=color, linewidth=1.4, linestyle="--",
            alpha=0.85, label="dut",
        )
        ax.set_title(name, fontsize=10)
        ax.set_xlim(0, 256)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _save_summary_histogram(
    ref_img: np.ndarray,
    dut_imgs: list[np.ndarray],
    out_path: Path,
) -> None:
    """Summary histogram: ref solid; every DUT as a thin translucent line in
    the same axes per channel. Lets the reader see at a glance whether DUTs
    cluster around ref or drift.
    """
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharey=True)
    for ax, (name, idx, color) in zip(axes, _RGB_CHANNELS):
        ax.hist(
            ref_img[..., idx].ravel(), bins=64, range=(0, 256),
            histtype="step", color=color, linewidth=1.8, label="ref",
        )
        for dut in dut_imgs:
            ax.hist(
                dut[..., idx].ravel(), bins=64, range=(0, 256),
                histtype="step", color=color, linewidth=0.6,
                linestyle="--", alpha=0.45,
            )
        ax.set_title(name, fontsize=10)
        ax.set_xlim(0, 256)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle(
        f"RGB histogram — REF (solid) vs all {len(dut_imgs)} DUT(s) (dashed)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _summary_table(
    aggregate_metrics: list[MetricResult],
    per_image_verdicts: list,
) -> str:
    """Per-metric pass-count across all DUT images.

    `aggregate_metrics` is used only to fix the metric order + direction + tol.
    The numbers in each row come from counting `within_tol` across all
    per-image verdicts.
    """
    n = len(per_image_verdicts)
    head = (
        "| Metric | Role | Direction | Tol % | Passed |\n"
        "|---|:--|:--|---:|:--:|\n"
    )
    rows = []
    for m in aggregate_metrics:
        passed = sum(
            1 for piv in per_image_verdicts
            for pm in piv.per_metric
            if pm.name == m.name and pm.within_tol
        )
        role = (
            '<span class="bad">blocking</span>' if m.is_blocking
            else '<span class="warn">advisory</span>'
        )
        count = f"{passed} / {n}"
        if passed < n and not m.is_blocking:
            count = f'<span class="warn">{count}</span>'
        elif passed < n:
            count = f'<span class="bad">{count}</span>'
        rows.append(
            f"| {m.name} "
            f"| {role} "
            f"| {m.direction.replace('_', ' ')} "
            f"| {_fmt(m.tolerance_pct, 1)} "
            f"| {count} |"
        )
    return head + "\n".join(rows)


def _per_dut_table_with_flags(metric_results: list[MetricResult]) -> str:
    """Per-image table: ref / dut / Δ / % / within-tol flag for each metric.

    `metric_results` here are this image's MetricResults (dut_agg ==
    single-image value), so we can read `within_tol` directly.
    """
    head = (
        "| Metric | Ref | DUT | Δ | % of ref | Tol % | Within tol? |\n"
        "|---|---:|---:|---:|---:|---:|:--:|\n"
    )
    rows = []
    for m in metric_results:
        name = m.name if m.is_blocking else f'{m.name} <span class="sev">(advisory)</span>'
        rows.append(
            f"| {name} "
            f"| {_fmt(m.ref)} "
            f"| {_fmt(m.dut_agg)} "
            f"| {_fmt(m.delta)} "
            f"| {_fmt(m.pct_of_ref, 1)} "
            f"| {_fmt(m.tolerance_pct, 1)} "
            f"| {_within_glyph(m)} |"
        )
    return head + "\n".join(rows)


def _deviation_phrase(m: MetricResult) -> str:
    """How far outside tolerance, in the units of the metric.

    `% of ref` is reported for ratio-style metrics but not for closer_to_ref
    ones: a*/b* are signed and sit near zero, so a ratio there is noise (an
    a* of 0.17 → -1.85 reads as "-1006% of ref"). The tolerance is an absolute
    band around the reference, so state the band.
    """
    if m.direction == "closer_to_ref":
        allowed = abs(m.ref) * m.tolerance_pct / 100.0
        return f"Δ {m.delta:+.2f} (allowed ±{_fmt(allowed)})"
    return f"Δ {m.delta:+.2f} ({_fmt(m.pct_of_ref, 1)}% of ref, tol {_fmt(m.tolerance_pct, 1)}%)"


_COLOUR_METRICS = frozenset({"L", "a*", "b*", "R", "G", "B"})


def _advisory_rationale(offenders: list[MetricResult]) -> str:
    """Why the listed advisory metrics are not grounds for rejection."""
    reasons = []
    if any(m.name in _COLOUR_METRICS for m in offenders):
        reasons.append(
            "colour and exposure deviations are correctable downstream "
            "(CCM / white balance)"
        )
    if any(m.name == "noise" for m in offenders):
        reasons.append(
            "a raised noise floor is a matter of degree rather than an optical "
            "defect"
        )
    if not reasons:
        return "These metrics are configured as advisory."
    clause = " and ".join(reasons)
    tail = "it is" if len(offenders) == 1 else "they are"
    return f"{clause[0].upper()}{clause[1:]}, so {tail} reported for review."


def _advisory_block(metric_results: list[MetricResult]) -> str | None:
    """Amber callout listing the advisory (colour / exposure / noise) metrics
    outside tolerance. Returns None when there is nothing to warn about.
    """
    offenders = [m for m in metric_results if m.warns]
    if not offenders:
        return None
    items = "; ".join(
        f"<strong>{m.name}</strong> {_deviation_phrase(m)}" for m in offenders
    )
    return (
        '<div class="advisory">'
        f'<strong>Warning — does not affect the verdict.</strong> '
        f'{len(offenders)} advisory metric(s) outside tolerance: {items}. '
        f'{_advisory_rationale(offenders)} '
        f'Only sharpness can fail a unit.'
        '</div>'
    )


def _preprocess_tag(preprocessed: bool) -> str:
    return f" — preprocessed (`{PREPROCESS_MODE}`)" if preprocessed else ""


def render_markdown(
    result: EvaluationResult,
    assets_dir: Path,
) -> str:
    v = result.verdict
    banner_label, banner_text = VERDICT_BANNERS[v.verdict]
    timestamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    assets_dir.mkdir(parents=True, exist_ok=True)
    log.info("Rendering report assets in %s", assets_dir)

    # Summary RGB histogram (all DUT vs ref).
    summary_hist = assets_dir / "summary_hist.png"
    _save_summary_histogram(
        result.inputs.ref_image, result.inputs.dut_images, summary_hist
    )

    # Per-DUT side-by-side + per-DUT histogram.
    pair_paths: list[Path] = []
    hist_paths: list[Path] = []
    for i, (dut_path, dut_img) in enumerate(tqdm(
        list(zip(result.inputs.dut_paths, result.inputs.dut_images)),
        desc="Rendering report assets", unit="img",
    )):
        pair_png = assets_dir / f"dut_{i:03d}_pair.png"
        hist_png = assets_dir / f"dut_{i:03d}_hist.png"
        _save_side_by_side(
            result.inputs.ref_image, dut_img,
            result.inputs.ref_path.name, dut_path.name, pair_png,
        )
        _save_pair_histogram(result.inputs.ref_image, dut_img, hist_png)
        pair_paths.append(pair_png)
        hist_paths.append(hist_png)

    md_lines: list[str] = []
    md_lines.append("# Camera QC Report")
    md_lines.append(f'<div class="banner {v.verdict}">{banner_label} &mdash; {banner_text}</div>')
    md_lines.append("")
    md_lines.append(f"- **Generated:** {timestamp}")
    md_lines.append(f"- **Scene preset:** `{result.scene}`")
    md_lines.append(f"- **Reference:** `{result.inputs.ref_path.name}`"
                    f"{_preprocess_tag(result.inputs.preprocess_ref)}")
    md_lines.append(f"- **DUT images:** {len(result.inputs.dut_paths)}"
                    f"{_preprocess_tag(result.inputs.preprocess_dut)}")
    md_lines.append("")
    if result.inputs.preprocess_asymmetric:
        processed = "reference" if result.inputs.preprocess_ref else "DUT"
        raw = "DUT" if result.inputs.preprocess_ref else "reference"
        md_lines.append(
            '<div class="disclaimer">'
            f'<strong>Warning — asymmetric preprocessing.</strong> '
            f'The {processed} went through the <code>{PREPROCESS_MODE}</code> '
            f'capture pipeline while the {raw} did not. The metric deltas below '
            f'therefore mix the processing difference with the sensor '
            f'difference and should not be read as a sensor-only comparison.'
            '</div>'
        )
        md_lines.append("")
    md_lines.append(
        '<div class="disclaimer">'
        '<strong>Disclaimer — preliminary thresholds.</strong> '
        'The pass/marginal/fail thresholds used in this report are preliminary '
        'and should be tuned against a batch of known-good DUTs before this '
        'report is used as a standalone acceptance decision.'
        '</div>'
    )
    md_lines.append("")
    md_lines.append("## Summary (per metric, across all DUT images)")
    md_lines.append(_summary_table(v.per_metric, result.per_image_verdicts))
    md_lines.append("")
    n_dut = len(result.per_image_verdicts)
    ge = _at_or_above_ref(result.per_image_verdicts)
    rc = _roll_call(result.per_image_verdicts)
    md_lines.append(
        f"- **Pass / fail:** {rc['pass']} pass, {rc['fail']} fail"
        + (f", {rc['marginal']} marginal" if rc["marginal"] else "")
        + f" of {n_dut} DUT image(s)")
    md_lines.append(
        f"- **At or above reference sharpness:** {ge} of {n_dut} "
        f"({ge / n_dut * 100:.1f}%) — stricter than PASS, which allows a drop "
        f"of up to the tolerance")
    md_lines.append("")
    advisory_block = _advisory_block(v.per_metric)
    if advisory_block:
        md_lines.append(advisory_block)
        md_lines.append("")
    md_lines.append(f"![rgb histogram]({summary_hist.name})")
    md_lines.append("")
    md_lines.append('<p class="legend">'
                    'Role key: <em>blocking</em> — sharpness; outside tolerance means FAIL. '
                    '<em>advisory</em> — colour / exposure / noise; outside tolerance is '
                    'highlighted as a warning and never fails the unit.<br>'
                    'Direction key: <em>higher better</em> — outside tolerance means it dropped >tol% below ref. '
                    '<em>lower better</em> — outside tolerance means it rose >tol% above ref. '
                    '<em>closer to ref</em> — outside tolerance means |Δ| exceeds tol% of ref.'
                    '</p>')
    md_lines.append("")
    md_lines.append("## Per-DUT-image comparisons")
    for i, dut_path in enumerate(result.inputs.dut_paths):
        piv = result.per_image_verdicts[i]
        md_lines.append(f"### DUT {i+1}: `{dut_path.name}`")
        advisory_note = (
            f', {piv.n_advisory_outside_tol}/{piv.n_advisory} advisory'
            if piv.n_advisory_outside_tol else ''
        )
        md_lines.append(
            _override_banner(
                piv,
                f' Measured: {piv.n_outside_tol}/{piv.n_blocking} blocking '
                f'metrics outside tolerance{advisory_note}.')
            or (f'<div class="banner small {piv.verdict}">'
                f'{piv.verdict.upper()} — {PER_IMAGE_BANNERS[piv.verdict]} '
                f'({piv.n_outside_tol}/{piv.n_blocking} blocking metrics '
                f'outside tolerance{advisory_note})'
                f'</div>')
        )
        md_lines.append(f"![ref vs dut]({pair_paths[i].name})")
        md_lines.append("")
        md_lines.append(
            f'<p class="conclusion"><strong>Conclusion:</strong> '
            f'{narrate(result.ref_metrics, result.dut_metrics[i])}</p>'
        )
        md_lines.append("")
        md_lines.append(_per_dut_table_with_flags(piv.per_metric))
        md_lines.append("")
        per_image_advisory = _advisory_block(piv.per_metric)
        if per_image_advisory:
            md_lines.append(per_image_advisory)
            md_lines.append("")
        md_lines.append(f"![rgb histogram]({hist_paths[i].name})")
        md_lines.append("")

    return "\n".join(md_lines)


SIMPLE_METRIC = "sharpness"


def _metric_by_name(metric_results: list[MetricResult], name: str) -> MetricResult:
    for m in metric_results:
        if m.name == name:
            return m
    raise KeyError(
        f"Metric {name!r} missing from the verdict. Present: "
        f"{[m.name for m in metric_results]}"
    )


def _roll_call(per_image_verdicts: list) -> dict[str, int]:
    counts = {"pass": 0, "marginal": 0, "fail": 0}
    for piv in per_image_verdicts:
        counts[piv.verdict] += 1
    return counts


def _override_banner(piv, extra: str = "") -> str | None:
    """Banner for a verdict that was forced regardless of the measurements.

    Stating the metric alongside a FAIL it did not cause reads as a bug, so
    the reason leads and the measurement is explicitly labelled as passing.
    """
    if not piv.override_reason:
        return None
    return (
        f'<div class="banner small fail">'
        f'FAIL — {piv.override_reason}. '
        f'This is a non-optical rejection: the optical measurement is within '
        f'tolerance and did not cause it.{extra}'
        f'</div>'
    )


def _at_or_above_ref(per_image_verdicts: list) -> int:
    """How many DUT images are at least as sharp as the reference.

    Distinct from PASS, which allows a drop of up to the tolerance: this is
    the stricter "not softer than the reference at all" count.
    """
    total = 0
    for piv in per_image_verdicts:
        m = _metric_by_name(piv.per_metric, SIMPLE_METRIC)
        total += m.dut_agg >= m.ref
    return total


def _headcount_table(per_image_verdicts: list) -> str:
    """Opening summary: how many images, how many passed / failed, and how
    many are at or above the reference outright."""
    n = len(per_image_verdicts)
    counts = _roll_call(per_image_verdicts)
    rows = [
        f"| **Total DUT images** | **{n}** | **100.0%** |",
        f'| <span class="ok">PASS</span> | {counts["pass"]} '
        f'| {counts["pass"] / n * 100:.1f}% |',
    ]
    # MARGINAL is unreachable at the default thresholds; only show the row when
    # a config actually produced one, so the common report stays two lines.
    if counts["marginal"]:
        rows.append(
            f'| <span class="warn">MARGINAL</span> | {counts["marginal"]} '
            f'| {counts["marginal"] / n * 100:.1f}% |'
        )
    rows.append(
        f'| <span class="bad">FAIL</span> | {counts["fail"]} '
        f'| {counts["fail"] / n * 100:.1f}% |'
    )
    ge = _at_or_above_ref(per_image_verdicts)
    rows.append(
        f'| &ge; reference sharpness | {ge} | {ge / n * 100:.1f}% |'
    )
    head = "| Result | Images | Share |\n|---|---:|---:|\n"
    return head + "\n".join(rows)


def _verdict_cell(verdict: str) -> str:
    css = {"pass": "ok", "marginal": "warn", "fail": "bad"}[verdict]
    return f'<span class="{css}">{verdict.upper()}</span>'


def _sharpness_table(
    dut_paths: list[Path],
    per_image_verdicts: list,
) -> str:
    """One row per DUT image: name, ref value, DUT value, pass/fail.

    A double-dagger marks a unit failed on non-optical grounds, so a FAIL
    sitting beside a perfectly good sharpness figure reads as intentional.
    """
    head = (
        "| # | DUT image | Ref sharpness | DUT sharpness | % of ref "
        "| Tol % | Result |\n"
        "|---:|---|---:|---:|---:|---:|:--:|\n"
    )
    rows = []
    for i, (path, piv) in enumerate(zip(dut_paths, per_image_verdicts), start=1):
        m = _metric_by_name(piv.per_metric, SIMPLE_METRIC)
        # A forced failure keeps its real measurement in the row, so the
        # marker is what stops a FAIL beside a healthy sharpness reading from
        # looking like a mistake.
        dagger = ' <span class="bad">‡</span>' if piv.override_reason else ""
        verdict_cell = _verdict_cell(piv.verdict) + dagger
        rows.append(
            f"| {i} "
            f"| `{path.name}` "
            f"| {_fmt(m.ref)} "
            f"| {_fmt(m.dut_agg)} "
            f"| {_fmt(m.pct_of_ref, 1)} "
            f"| {_fmt(m.tolerance_pct, 1)} "
            f"| {verdict_cell} |"
        )
    return head + "\n".join(rows)


def render_markdown_simplified(
    result: EvaluationResult,
    assets_dir: Path,
) -> str:
    """Sharpness-only report: headcount summary, one table row per DUT image,
    then the ref-vs-DUT image pairs.

    No histograms and no per-metric colour breakdown — sharpness is the only
    metric that can fail a unit, so everything else is omitted rather than
    computed and ignored.
    """
    v = result.verdict
    banner_label, banner_text = VERDICT_BANNERS[v.verdict]
    timestamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    assets_dir.mkdir(parents=True, exist_ok=True)
    log.info("Rendering report assets in %s", assets_dir)

    pair_paths: list[Path] = []
    for i, (dut_path, dut_img) in enumerate(tqdm(
        list(zip(result.inputs.dut_paths, result.inputs.dut_images)),
        desc="Rendering report assets", unit="img",
    )):
        pair_png = assets_dir / f"dut_{i:03d}_pair.png"
        _save_side_by_side(
            result.inputs.ref_image, dut_img,
            result.inputs.ref_path.name, dut_path.name, pair_png,
        )
        pair_paths.append(pair_png)

    agg = _metric_by_name(v.per_metric, SIMPLE_METRIC)

    md: list[str] = []
    md.append("# Camera QC Report — sharpness")
    md.append(f'<div class="banner {v.verdict}">{banner_label} &mdash; {banner_text}</div>')
    md.append("")
    md.append("## Summary")
    md.append("")
    md.append(_headcount_table(result.per_image_verdicts))
    md.append("")
    md.append(f"- **Generated:** {timestamp}")
    md.append(f"- **Scene preset:** `{result.scene}`")
    md.append(f"- **Reference:** `{result.inputs.ref_path.name}`"
              f"{_preprocess_tag(result.inputs.preprocess_ref)}"
              f" — sharpness {_fmt(agg.ref)}")
    md.append(f"- **DUT images:** {len(result.inputs.dut_paths)}"
              f"{_preprocess_tag(result.inputs.preprocess_dut)}")
    md.append(f"- **Pass criterion:** DUT sharpness \u2265 "
              f"{100 - agg.tolerance_pct:.0f}% of reference "
              f"({_fmt(agg.ref * (1 - agg.tolerance_pct / 100))})")
    md.append(f"- **Batch aggregate:** {_fmt(agg.dut_agg)} "
              f"({_fmt(agg.pct_of_ref, 1)}% of ref, {v.aggregation} "
              f"across {v.n_dut} image(s))")
    md.append("")
    if result.inputs.preprocess_asymmetric:
        processed = "reference" if result.inputs.preprocess_ref else "DUT"
        raw = "DUT" if result.inputs.preprocess_ref else "reference"
        md.append(
            '<div class="disclaimer">'
            f'<strong>Warning — asymmetric preprocessing.</strong> '
            f'The {processed} went through the <code>{PREPROCESS_MODE}</code> '
            f'capture pipeline while the {raw} did not. The sharpness deltas '
            f'below therefore mix the processing difference with the sensor '
            f'difference and should not be read as a sensor-only comparison.'
            '</div>'
        )
        md.append("")
    md.append(
        '<div class="disclaimer">'
        '<strong>Disclaimer — preliminary thresholds.</strong> '
        'The pass/fail threshold used in this report is preliminary and should '
        'be tuned against a batch of known-good DUTs before this report is used '
        'as a standalone acceptance decision. This is the simplified report: '
        'sharpness only. Colour, exposure and noise are advisory metrics that '
        'never fail a unit and were not measured — run with '
        '<code>--no-simplified</code> for the full pack.'
        '</div>'
    )
    md.append("")
    # Batch-specific annotations belong to whoever built the batch, not to
    # this generator: they stay in manifest.csv and are never rendered here.
    md.append("## Per-image results")
    md.append(_sharpness_table(result.inputs.dut_paths,
                               result.per_image_verdicts))
    md.append("")
    forced = [(i, p.name, result.per_image_verdicts[i - 1].override_reason)
              for i, p in enumerate(result.inputs.dut_paths, start=1)
              if result.per_image_verdicts[i - 1].override_reason]
    legend = ('Sharpness is the variance of the Laplacian — higher is sharper. '
              'A unit fails the <em>optical</em> check when its sharpness '
              'drops more than the tolerance below the reference; sharper '
              'than reference always passes that check.')
    if forced:
        listed = "; ".join(f"#{i} <code>{n}</code> — {r}" for i, n, r in forced)
        legend += (f'<br><span class="bad">‡</span> <strong>{len(forced)} '
                   f'unit(s) carry a FAIL that the sharpness column does not '
                   f'explain.</strong> They passed the optical check and were '
                   f'rejected on non-optical grounds, so a high % of ref '
                   f'beside their FAIL is expected, not an error: {listed}.')
    md.append(f'<p class="legend">{legend}</p>')
    md.append("")
    md.append("## Per-DUT-image comparisons")
    for i, dut_path in enumerate(result.inputs.dut_paths):
        piv = result.per_image_verdicts[i]
        m = _metric_by_name(piv.per_metric, SIMPLE_METRIC)
        md.append(f"### DUT {i+1}: `{dut_path.name}`")
        measured = (f' Sharpness {_fmt(m.dut_agg)} vs ref {_fmt(m.ref)} '
                    f'({_fmt(m.pct_of_ref, 1)}% of ref).')
        md.append(
            _override_banner(piv, measured)
            or (f'<div class="banner small {piv.verdict}">'
                f'{piv.verdict.upper()} — sharpness {_fmt(m.dut_agg)} vs ref '
                f'{_fmt(m.ref)} ({_fmt(m.pct_of_ref, 1)}% of ref, '
                f'tolerance {_fmt(m.tolerance_pct, 1)}%)'
                f'</div>')
        )
        md.append(f"![ref vs dut]({pair_paths[i].name})")
        md.append("")

    return "\n".join(md)


def write_pdf(result: EvaluationResult, out_path: Path, keep_md: bool = False) -> Path:
    """Render MD + assets in a temp dir, convert to PDF at `out_path`."""
    import markdown as md_lib
    from weasyprint import HTML, CSS as WeasyCSS

    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    work_root = (
        out_path.parent / f".{out_path.stem}_work"
        if keep_md
        else Path(tempfile.mkdtemp(prefix="camera_qc_"))
    )
    work_root.mkdir(parents=True, exist_ok=True)
    assets_dir = work_root  # keep PNGs alongside the MD so relative paths resolve

    render = (render_markdown_simplified if result.simplified
              else render_markdown)
    md_text = render(result, assets_dir)
    md_path = work_root / "report.md"
    md_path.write_text(md_text, encoding="utf-8")

    html_body = md_lib.markdown(
        md_text,
        extensions=["tables", "extra"],
    )
    html_doc = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'></head>"
        f"<body>{html_body}</body></html>"
    )
    HTML(string=html_doc, base_url=str(work_root)).write_pdf(
        str(out_path),
        stylesheets=[WeasyCSS(string=CSS)],
    )

    if not keep_md:
        shutil.rmtree(work_root, ignore_errors=True)

    return out_path
