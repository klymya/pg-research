"""Combine every per-session batch into one document, with a cross-session
analysis at the front.

Each session batch answers "how did these modules measure at 6.02PM against
this reference". None of them answers the question that matters once you have
nine of them: *do the sessions agree with each other?* A module that ranks
sharpest at 5.07PM and mid-pack at 6.31PM is telling you the measurement moves,
not the module.

The summary therefore leads with correlation between sessions, and the
per-session sections follow in chronological order, each split by reference
unit.

Usage:
    python -m camera_qc.combine_session_reports --root ./revc \
        --scene ceiling --out ./revc/combined_report.pdf
"""

from __future__ import annotations

import argparse
import datetime as _dt
import logging
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .evaluate import run_evaluation
from .report import (
    CSS, _fmt, _save_side_by_side, _verdict_cell,
)


log = logging.getLogger(__name__)

SIMPLE_METRIC = "sharpness"

EXTRA_CSS = """
.grid { font-size: 8.5pt; }
.grid td, .grid th { padding: 3px 5px; }
.hi { background: #e6f4ea; }
.mid { background: #fff8e1; }
.lo { background: #fdecea; }
.finding { background: #eef2fb; border-left: 3px solid #4466aa; padding: 8px 12px;
           margin: 0.6em 0; font-size: 9.5pt; }
h4 { font-size: 10pt; margin: 1em 0 0.3em 0; }
"""


@dataclass
class Batch:
    name: str
    session: str
    ref_serial: int
    result: object          # EvaluationResult

    @property
    def sharpness_by_unit(self) -> dict[int, float]:
        return {
            int(p.stem.split("-")[2]): m[SIMPLE_METRIC]
            for p, m in zip(self.result.inputs.dut_paths, self.result.dut_metrics)
        }

    @property
    def verdict_by_unit(self) -> dict[int, str]:
        return {
            int(p.stem.split("-")[2]): v.verdict
            for p, v in zip(self.result.inputs.dut_paths,
                            self.result.per_image_verdicts)
        }


def spearman(a: list[float], b: list[float]) -> float:
    """Rank correlation. Implemented here rather than pulled from scipy so the
    package keeps its existing dependency set. Ties get average ranks.
    """
    if len(a) < 3:
        return float("nan")

    def rank(v: list[float]) -> np.ndarray:
        arr = np.asarray(v, dtype=float)
        order = arr.argsort()
        ranks = np.empty(len(arr), dtype=float)
        ranks[order] = np.arange(len(arr), dtype=float)
        # Average the ranks of tied values, or identical readings would be
        # ordered arbitrarily and invent a correlation that is not there.
        for value in np.unique(arr):
            mask = arr == value
            if mask.sum() > 1:
                ranks[mask] = ranks[mask].mean()
        return ranks

    ra, rb = rank(a), rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def load_batches(root: Path, scene: str, simplified: bool) -> list[Batch]:
    from .build_session_batches import session_minutes

    dirs = [d for d in sorted(root.iterdir())
            if d.is_dir() and (d / "ref").is_dir() and (d / "dut").is_dir()]
    if not dirs:
        raise SystemExit(f"No batch folders (with ref/ and dut/) under {root}")

    batches: list[Batch] = []
    for d in tqdm(dirs, desc="Evaluating batches", unit="batch"):
        session, _, ref = d.name.rpartition("_ref")
        if not ref.isdigit():
            log.warning("%s — name is not <session>_ref<serial>, skipping", d.name)
            continue
        result = run_evaluation(d / "ref", d / "dut", scene, simplified=simplified)
        batches.append(Batch(d.name, session, int(ref), result))

    batches.sort(key=lambda b: (session_minutes(b.session) or 0, b.ref_serial))
    log.info("Loaded %d batch(es) across %d session(s)",
             len(batches), len({b.session for b in batches}))
    return batches


def _shade(v: float, hi: float, lo: float) -> str:
    return "hi" if v >= hi else ("lo" if v < lo else "mid")


def session_level_table(batches: list[Batch], sessions: list[str]) -> str:
    """Median sharpness of the units measured in each session.

    A session that sits far from the others moved every module at once, which
    is a property of the capture conditions and not of the hardware.
    """
    rows = []
    per_session = {}
    for s in sessions:
        vals = []
        for b in batches:
            if b.session == s:
                vals.extend(b.sharpness_by_unit.values())
                vals.append(b.result.ref_metrics[SIMPLE_METRIC])
        per_session[s] = float(np.median(vals))
    overall = float(np.median(list(per_session.values())))
    for s in sessions:
        v = per_session[s]
        pct = v / overall * 100
        rows.append(
            f'| {s} | {v:.0f} | <span class="{_shade(pct, 90, 80)}">{pct:.0f}%</span> |'
        )
    head = ("| Session | Median sharpness (all units) | vs all-session median |\n"
            "|---|---:|---:|\n")
    return head + "\n".join(rows), per_session, overall


def correlation_matrix(batches: list[Batch], sessions: list[str]) -> tuple[str, dict]:
    """Session x session rank correlation over the per-unit sharpness values.

    Rank rather than absolute: the question is whether the sessions agree on
    which modules are the sharp ones, independently of the overall level
    drifting between sessions.
    """
    by_session: dict[str, dict[int, float]] = {}
    for s in sessions:
        merged: dict[int, float] = {}
        for b in batches:
            if b.session == s:
                merged.update(b.sharpness_by_unit)
                merged[b.ref_serial] = b.result.ref_metrics[SIMPLE_METRIC]
        by_session[s] = merged

    pairs: dict[tuple[str, str], float] = {}
    head = "| | " + " | ".join(sessions) + " |\n|---|" + "---:|" * len(sessions) + "\n"
    rows = []
    for a in sessions:
        cells = []
        for b in sessions:
            if a == b:
                cells.append("—")
                continue
            shared = sorted(set(by_session[a]) & set(by_session[b]))
            r = spearman([by_session[a][u] for u in shared],
                         [by_session[b][u] for u in shared])
            pairs[(a, b)] = r
            cells.append("n/a" if np.isnan(r)
                         else f'<span class="{_shade(r, 0.7, 0.3)}">{r:+.2f}</span>')
        rows.append(f"| **{a}** | " + " | ".join(cells) + " |")
    return head + "\n".join(rows), pairs


def unit_stability_table(batches: list[Batch], sessions: list[str]) -> str:
    """Each unit's sharpness across sessions, and how much it moved."""
    by_unit: dict[int, dict[str, float]] = {}
    for b in batches:
        for u, v in b.sharpness_by_unit.items():
            by_unit.setdefault(u, {})[b.session] = v
        by_unit.setdefault(b.ref_serial, {})[b.session] = \
            b.result.ref_metrics[SIMPLE_METRIC]

    head = ("| Unit | " + " | ".join(sessions) + " | median | spread |\n"
            "|---|" + "---:|" * (len(sessions) + 2) + "\n")
    rows = []
    for u in sorted(by_unit):
        vals = by_unit[u]
        cells = [f"{vals[s]:.0f}" if s in vals else "—" for s in sessions]
        v = np.array(list(vals.values()))
        spread = v.max() / v.min()
        rows.append(
            f"| **#{u}** | " + " | ".join(cells) +
            f' | {np.median(v):.0f} '
            f'| <span class="{_shade(2 - spread, 1.3, 1.0)}">{spread:.2f}x</span> |'
        )
    return head + "\n".join(rows)


def verdict_stability_table(batches: list[Batch], sessions: list[str],
                            refs: list[int]) -> tuple[str, dict]:
    """How often each unit passes, per reference, across the sessions.

    A unit that passes in some sessions and fails in others has no stable
    verdict — the number below is the count, not a quality score.
    """
    tally: dict[tuple[int, int], list[str]] = {}
    for b in batches:
        for u, verdict in b.verdict_by_unit.items():
            tally.setdefault((u, b.ref_serial), []).append(verdict)

    units = sorted({u for u, _r in tally})
    head = ("| Unit | " + " | ".join(f"vs #{r}" for r in refs) + " | stable? |\n"
            "|---|" + "---:|" * len(refs) + ":--|\n")
    rows = []
    unstable = []
    for u in units:
        cells, flips = [], False
        for r in refs:
            vs = tally.get((u, r), [])
            if not vs:
                cells.append("—")
                continue
            n_pass = sum(1 for v in vs if v == "pass")
            if 0 < n_pass < len(vs):
                flips = True
            shade = "hi" if n_pass == len(vs) else ("lo" if n_pass == 0 else "mid")
            cells.append(f'<span class="{shade}">{n_pass}/{len(vs)} pass</span>')
        label = ('<span class="mid">flips by session</span>' if flips
                 else "consistent")
        if flips:
            unstable.append(u)
        rows.append(f"| **#{u}** | " + " | ".join(cells) + f" | {label} |")
    return head + "\n".join(rows), {"unstable": unstable, "units": units}


def reference_disagreement(batches: list[Batch]) -> tuple[str, dict]:
    """Verdicts for the identical image under each reference.

    Same file, same session, different yardstick: every disagreement here is
    produced purely by the choice of reference unit.
    """
    seen: dict[tuple[str, int], dict[int, str]] = {}
    for b in batches:
        for u, verdict in b.verdict_by_unit.items():
            seen.setdefault((b.session, u), {})[b.ref_serial] = verdict

    both = {k: v for k, v in seen.items() if len(v) > 1}
    disagree = {k: v for k, v in both.items() if len(set(v.values())) > 1}
    rows = []
    for (session, u), verdicts in sorted(disagree.items()):
        cells = " | ".join(
            _verdict_cell(verdicts[r]) for r in sorted(verdicts))
        rows.append(f"| {session} | **#{u}** | {cells} |")
    refs = sorted({r for v in both.values() for r in v})
    head = ("| Session | Unit | " + " | ".join(f"vs #{r}" for r in refs) + " |\n"
            "|---|---|" + ":--:|" * len(refs) + "\n")
    table = head + "\n".join(rows) if rows else None
    return table, {"compared": len(both), "disagree": len(disagree)}


def per_image_table(batch: Batch) -> str:
    head = ("| # | DUT image | Unit | Ref sharpness | DUT sharpness "
            "| % of ref | Result |\n|---:|---|---:|---:|---:|---:|:--:|\n")
    rows = []
    for i, (p, m, piv) in enumerate(zip(
            batch.result.inputs.dut_paths, batch.result.dut_metrics,
            batch.result.per_image_verdicts), start=1):
        ref = batch.result.ref_metrics[SIMPLE_METRIC]
        rows.append(
            f"| {i} | `{p.name}` | #{int(p.stem.split('-')[2])} "
            f"| {_fmt(ref)} | {_fmt(m[SIMPLE_METRIC])} "
            f"| {_fmt(m[SIMPLE_METRIC] / ref * 100, 1)} "
            f"| {_verdict_cell(piv.verdict)} |")
    return head + "\n".join(rows)


def render(batches: list[Batch], assets: Path, with_images: bool,
           scene: str) -> str:
    sessions = sorted({b.session for b in batches},
                      key=lambda s: [b.session for b in batches].index(s))
    refs = sorted({b.ref_serial for b in batches})
    timestamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    n_cmp = sum(len(b.result.dut_metrics) for b in batches)
    n_pass = sum(1 for b in batches for v in b.result.per_image_verdicts
                 if v.verdict == "pass")

    level_table, per_session, overall = session_level_table(batches, sessions)
    corr_table, pairs = correlation_matrix(batches, sessions)
    stability = unit_stability_table(batches, sessions)
    verdict_table, vinfo = verdict_stability_table(batches, sessions, refs)
    disagree_table, dinfo = reference_disagreement(batches)

    finite = [r for r in pairs.values() if not np.isnan(r)]
    median_r = float(np.median(finite)) if finite else float("nan")
    outliers = [s for s in sessions
                if per_session[s] / overall * 100 < 80]

    md: list[str] = []
    md.append("# Camera QC — combined session report")
    md.append(f"*{len(sessions)} sessions &times; {len(refs)} reference units "
              f"= {len(batches)} batches, {n_cmp} comparisons*")
    md.append("")
    md.append(f"- **Generated:** {timestamp}")
    md.append(f"- **Scene preset:** `{scene}`")
    md.append(f"- **References:** " + ", ".join(f"#{r}" for r in refs))
    md.append(f"- **Overall:** {n_pass} pass / {n_cmp - n_pass} fail "
              f"across all batches")
    md.append("")

    md.append("## Summary")
    md.append("")
    md.append("### Do the sessions agree?")
    md.append("")
    md.append(corr_table)
    md.append("")
    md.append(
        '<p class="legend">Spearman rank correlation of per-unit sharpness '
        'between each pair of sessions. <strong>+1.00</strong> means the two '
        'sessions rank the modules identically — the measurement is '
        'repeatable even if the absolute level drifts. Near <strong>0</strong> '
        'means the ranking is unrelated, i.e. the numbers are dominated by '
        'capture conditions rather than by the modules.</p>')
    md.append("")
    verdict = ("strong" if median_r >= 0.7 else
               "moderate" if median_r >= 0.3 else "weak")
    md.append(
        f'<div class="finding"><strong>Median pairwise correlation: '
        f'{median_r:+.2f} ({verdict}).</strong> '
        + ("The sessions largely agree on which modules are sharper, so "
           "relative ranking is trustworthy even where absolute values move."
           if median_r >= 0.7 else
           "The sessions do not consistently agree on which modules are "
           "sharper. Absolute pass/fail thresholds cannot be trusted while "
           "the ranking itself moves between captures.")
        + '</div>')
    md.append("")

    md.append("### Session level")
    md.append("")
    md.append(level_table)
    md.append("")
    if outliers:
        md.append(
            f'<div class="finding"><strong>Outlier session(s): '
            f'{", ".join(outliers)}.</strong> Every unit measured low here at '
            f'once. A whole-session shift is a property of the capture — '
            f'lighting, focus, scene content — not of the modules, so failures '
            f'confined to these sessions should not be read as hardware '
            f'defects.</div>')
        md.append("")

    md.append("### Per-unit stability across sessions")
    md.append("")
    md.append(stability)
    md.append("")
    md.append('<p class="legend">"Spread" is max / min for that unit across '
              'all sessions. With a 10% tolerance, any unit whose own spread '
              'exceeds 1.10x cannot be given a stable verdict from a single '
              'capture.</p>')
    md.append("")

    md.append("### Verdict stability")
    md.append("")
    md.append(verdict_table)
    md.append("")
    if vinfo["unstable"]:
        md.append(
            f'<div class="finding"><strong>'
            f'{len(vinfo["unstable"])} of {len(vinfo["units"])} units change '
            f'verdict depending on the session: '
            f'{", ".join("#" + str(u) for u in vinfo["unstable"])}.</strong> '
            f'For these, a single session decides pass or fail by chance.</div>')
        md.append("")

    md.append("### Effect of the reference choice")
    md.append("")
    if disagree_table:
        md.append(
            f'<div class="finding"><strong>{dinfo["disagree"]} of '
            f'{dinfo["compared"]} comparisons ({dinfo["disagree"] / dinfo["compared"] * 100:.0f}%) '
            f'flip verdict purely because the reference unit changed.</strong> '
            f'Same image, same session, opposite result.</div>')
        md.append("")
        md.append(disagree_table)
    else:
        md.append("Both references agree on every comparison.")
    md.append("")

    md.append("## Sessions")
    for session in sessions:
        md.append(f"### {session}")
        for b in [x for x in batches if x.session == session]:
            r = b.result
            n_p = sum(1 for v in r.per_image_verdicts if v.verdict == "pass")
            md.append(f"#### Reference #{b.ref_serial} "
                      f"&mdash; `{r.inputs.ref_path.name}`")
            md.append(
                f'<div class="banner small {"pass" if n_p == len(r.dut_metrics) else "fail"}">'
                f'{n_p} pass / {len(r.dut_metrics) - n_p} fail &mdash; '
                f'reference sharpness {_fmt(r.ref_metrics[SIMPLE_METRIC])}, '
                f'pass floor {_fmt(r.ref_metrics[SIMPLE_METRIC] * 0.9)}</div>')
            md.append(per_image_table(b))
            md.append("")
            if with_images:
                for i, (p, img) in enumerate(zip(r.inputs.dut_paths,
                                                 r.inputs.dut_images)):
                    png = assets / f"{b.name}_{i:02d}.png"
                    _save_side_by_side(r.inputs.ref_image, img,
                                       r.inputs.ref_path.name, p.name, png)
                    md.append(f"![{p.name}]({png.name})")
                    md.append("")
    return "\n".join(md)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="combine-session-reports",
        description="Combine per-session batches into one report with a "
                    "cross-session correlation summary.",
    )
    p.add_argument("--root", type=Path, required=True,
                   help="Folder holding the <session>_ref<serial> batches.")
    p.add_argument("--out", type=Path, required=True, help="Output PDF path.")
    p.add_argument("--scene", default="ceiling",
                   choices=("chessboard", "ceiling"))
    p.add_argument("--no-images", action="store_true",
                   help="Tables and analysis only — much smaller and faster.")
    p.add_argument("--full-metrics", action="store_true",
                   help="Measure the full metric pack instead of sharpness "
                        "only. The combined report shows sharpness either way; "
                        "this only changes the verdicts.")
    p.add_argument("--keep-md", action="store_true",
                   help="Keep the intermediate Markdown + assets.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-5s  %(message)s")
    for noisy in ("weasyprint", "fontTools", "PIL", "matplotlib",
                  "camera_qc.loader", "camera_qc.evaluate", "camera_qc.report"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    import markdown as md_lib
    from weasyprint import HTML, CSS as WeasyCSS

    root = args.root.expanduser()
    out = args.out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    batches = load_batches(root, args.scene, simplified=not args.full_metrics)

    work = (out.parent / f".{out.stem}_work" if args.keep_md
            else Path(tempfile.mkdtemp(prefix="camera_qc_combined_")))
    work.mkdir(parents=True, exist_ok=True)

    log.info("Rendering combined report%s",
             "" if args.no_images else " (with image pairs)")
    md_text = render(batches, work, not args.no_images, args.scene)
    (work / "report.md").write_text(md_text, encoding="utf-8")

    html = md_lib.markdown(md_text, extensions=["tables", "extra"])
    HTML(string=f"<!DOCTYPE html><html><head><meta charset='utf-8'></head>"
                f"<body>{html}</body></html>",
         base_url=str(work)).write_pdf(
        str(out), stylesheets=[WeasyCSS(string=CSS + EXTRA_CSS)])

    if args.keep_md:
        log.info("Markdown kept at %s", work / "report.md")
    else:
        shutil.rmtree(work, ignore_errors=True)
    log.info("Combined report: %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
