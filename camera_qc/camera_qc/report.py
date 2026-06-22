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
from .verdict import MetricResult


log = logging.getLogger(__name__)

# R, G, B channel index + plot colour, used for the histogram panels.
_RGB_CHANNELS: tuple[tuple[str, int, str], ...] = (
    ("R", 0, "tab:red"),
    ("G", 1, "tab:green"),
    ("B", 2, "tab:blue"),
)


VERDICT_BANNERS: dict[str, tuple[str, str]] = {
    "pass": ("PASS", "DUT is same or better than reference."),
    "marginal": ("MARGINAL", "DUT is borderline — manual review recommended."),
    "fail": ("FAIL", "DUT is worse than reference on multiple metrics."),
}

PER_IMAGE_BANNERS: dict[str, str] = {
    "pass": "within tolerance on every metric.",
    "marginal": "one metric outside tolerance.",
    "fail": "two or more metrics outside tolerance.",
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


def _within_glyph(ok: bool) -> str:
    return "OK" if ok else "OUT"


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
        "| Metric | Direction | Tol % | Passed |\n"
        "|---|:--|---:|:--:|\n"
    )
    rows = []
    for m in aggregate_metrics:
        passed = sum(
            1 for piv in per_image_verdicts
            for pm in piv.per_metric
            if pm.name == m.name and pm.within_tol
        )
        rows.append(
            f"| {m.name} "
            f"| {m.direction.replace('_', ' ')} "
            f"| {_fmt(m.tolerance_pct, 1)} "
            f"| {passed} / {n} |"
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
        rows.append(
            f"| {m.name} "
            f"| {_fmt(m.ref)} "
            f"| {_fmt(m.dut_agg)} "
            f"| {_fmt(m.delta)} "
            f"| {_fmt(m.pct_of_ref, 1)} "
            f"| {_fmt(m.tolerance_pct, 1)} "
            f"| {_within_glyph(m.within_tol)} |"
        )
    return head + "\n".join(rows)


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
    md_lines.append(f"- **Reference:** `{result.inputs.ref_path.name}`")
    md_lines.append(f"- **DUT images:** {len(result.inputs.dut_paths)}")
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
    md_lines.append(f"![rgb histogram]({summary_hist.name})")
    md_lines.append("")
    md_lines.append('<p class="legend">'
                    'Direction key: <em>higher better</em> — DUT failing means it dropped >tol% below ref. '
                    '<em>lower better</em> — DUT failing means it rose >tol% above ref. '
                    '<em>closer to ref</em> — DUT failing means |Δ| exceeds tol% of ref.'
                    '</p>')
    md_lines.append("")
    md_lines.append("## Per-DUT-image comparisons")
    for i, dut_path in enumerate(result.inputs.dut_paths):
        piv = result.per_image_verdicts[i]
        md_lines.append(f"### DUT {i+1}: `{dut_path.name}`")
        md_lines.append(
            f'<div class="banner small {piv.verdict}">'
            f'{piv.verdict.upper()} — {PER_IMAGE_BANNERS[piv.verdict]} '
            f'({piv.n_outside_tol}/{len(piv.per_metric)} metrics outside tolerance)'
            f'</div>'
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
        md_lines.append(f"![rgb histogram]({hist_paths[i].name})")
        md_lines.append("")

    return "\n".join(md_lines)


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

    md_text = render_markdown(result, assets_dir)
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
