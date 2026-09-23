"""Generate the A4 floor-marker sheet for the detection-area data collection.

One black circle per A4 sheet, as large as a consumer printer can reliably put on
the page. Output is vector PDF, so it is resolution-independent — the printer
renders it at its own native dpi with no rasterisation loss.

The sheet carries three aids that matter for a measurement protocol:

- **Edge registration ticks** at the four edge midpoints, to line the sheet up
  with the chalk grid lines on the floor.
- **A 100 mm scale bar.** If the printer was left on "Fit to page" the circle
  comes out the wrong size and every pixels-per-metre figure derived from it is
  wrong. Checking the bar with a ruler catches that in five seconds.
- **The nominal diameter printed on the sheet**, so a stack of sheets from
  different print runs can never be silently mixed.

Usage:
    python make_marker_sheet.py                      # 1 page,  190 mm circle
    python make_marker_sheet.py --pages 150          # a full campaign, one job
    python make_marker_sheet.py --diameter 180       # override the size
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Circle

# ISO 216 A4, in millimetres.
A4_W_MM, A4_H_MM = 210.0, 297.0
MM_PER_INCH = 25.4

# Consumer inkjet and laser printers reserve an unprintable border. 5 mm covers
# most, 10 mm covers essentially all of them and still leaves the circle larger
# than the 180 mm the protocol was originally drafted around.
DEFAULT_MARGIN_MM = 10.0

SCALE_BAR_MM = 100.0


def build_page(fig, diameter_mm: float, margin_mm: float) -> None:
    """Draw one marker sheet onto `fig`, in millimetre coordinates."""
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, A4_W_MM)
    ax.set_ylim(0, A4_H_MM)
    ax.set_aspect("equal")
    ax.axis("off")

    cx, cy = A4_W_MM / 2, A4_H_MM / 2
    ax.add_patch(Circle((cx, cy), diameter_mm / 2, facecolor="black",
                        edgecolor="none", linewidth=0))

    # Registration ticks at the four edge midpoints. Kept thin and short so they
    # cannot be mistaken for a blob by the marker detector, which filters on area
    # and circularity.
    tick = 8.0
    for x0, y0, x1, y1 in (
        (cx, 0, cx, tick),                      # bottom
        (cx, A4_H_MM - tick, cx, A4_H_MM),      # top
        (0, cy, tick, cy),                      # left
        (A4_W_MM - tick, cy, A4_W_MM, cy),      # right
    ):
        ax.plot([x0, x1], [y0, y1], color="black", linewidth=0.6,
                solid_capstyle="butt")

    # Top: the instruction that decides whether the sheet is usable at all.
    ax.text(cx, A4_H_MM - 16, "PRINT AT 100 % — ACTUAL SIZE",
            ha="center", va="center", fontsize=11, fontweight="bold")
    ax.text(cx, A4_H_MM - 24,
            'Do NOT use "Fit to page", "Shrink to fit" or "Scale to fit".',
            ha="center", va="center", fontsize=8)

    # Bottom: scale bar with end caps, for the ruler check.
    bar_y = 30.0
    x0, x1 = cx - SCALE_BAR_MM / 2, cx + SCALE_BAR_MM / 2
    ax.plot([x0, x1], [bar_y, bar_y], color="black", linewidth=1.0)
    for x in (x0, x1):
        ax.plot([x, x], [bar_y - 2.5, bar_y + 2.5], color="black", linewidth=1.0)
    ax.text(cx, bar_y - 7,
            f"{SCALE_BAR_MM:.0f} mm — check with a ruler before printing the rest",
            ha="center", va="center", fontsize=8)

    ax.text(cx, 15, f"Circle diameter {diameter_mm:.0f} mm   ·   "
                    f"Detection Area Size Estimation — floor marker",
            ha="center", va="center", fontsize=7, color="0.35")


def max_diameter(margin_mm: float) -> float:
    """Largest circle that fits inside the printable area of the page."""
    return min(A4_W_MM, A4_H_MM) - 2 * margin_mm


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="make-marker-sheet",
        description="Generate the A4 floor-marker sheet (black circle) as a "
                    "vector PDF.",
    )
    p.add_argument("--out", type=Path, default=Path("floor_marker_A4.pdf"),
                   help="Output PDF path (default: floor_marker_A4.pdf).")
    p.add_argument("--margin", type=float, default=DEFAULT_MARGIN_MM,
                   metavar="MM",
                   help=f"Unprintable border to keep clear on every side "
                        f"(default: {DEFAULT_MARGIN_MM:.0f} mm). The circle is "
                        f"sized to fill what is left.")
    p.add_argument("--diameter", type=float, default=None, metavar="MM",
                   help="Override the circle diameter instead of maximising it.")
    p.add_argument("--pages", type=int, default=1, metavar="N",
                   help="Emit N identical pages, so a whole campaign prints as "
                        "one job (default: 1).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.margin < 0:
        raise SystemExit("--margin cannot be negative.")
    if args.pages < 1:
        raise SystemExit("--pages must be at least 1.")

    limit = max_diameter(args.margin)
    if limit <= 0:
        raise SystemExit(
            f"A margin of {args.margin:.0f} mm leaves no room on a "
            f"{A4_W_MM:.0f} mm wide page.")

    diameter = args.diameter if args.diameter is not None else limit
    if diameter > limit:
        raise SystemExit(
            f"A {diameter:.0f} mm circle does not fit inside a "
            f"{args.margin:.0f} mm margin — the maximum is {limit:.0f} mm. "
            f"Reduce --diameter, or lower --margin if the printer allows it.")
    if diameter <= 0:
        raise SystemExit("--diameter must be positive.")

    figsize = (A4_W_MM / MM_PER_INCH, A4_H_MM / MM_PER_INCH)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(args.out) as pdf:
        for _ in range(args.pages):
            fig = plt.figure(figsize=figsize)
            build_page(fig, diameter, args.margin)
            # No bbox_inches="tight" — it would crop the page and destroy the
            # A4 geometry the whole sheet depends on.
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Wrote {args.out}")
    print(f"  page      A4, {A4_W_MM:.0f} x {A4_H_MM:.0f} mm")
    print(f"  circle    {diameter:.1f} mm diameter"
          f"{'' if args.diameter is not None else ' (maximised)'}")
    print(f"  margin    {(A4_W_MM - diameter) / 2:.1f} mm left/right, "
          f"{(A4_H_MM - diameter) / 2:.1f} mm top/bottom")
    print(f"  pages     {args.pages}")
    print(f"  vector PDF — prints at the printer's own resolution")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
