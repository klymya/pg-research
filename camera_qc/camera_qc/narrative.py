"""One-line, human-readable conclusion for a single DUT image vs ref.

Mirrors the prose style used in baseline_comparison_test3.ipynb section
captions, e.g.:
    sharpness 89.7 / 95.5% (avg ~92%); noise 94% of ref (cleaner);
    channels 3.0–3.7% darker; color stable.
"""

from __future__ import annotations

from typing import Mapping


def _pct(dut: float, ref: float) -> float:
    if ref == 0:
        return float("nan")
    return dut / ref * 100.0


def narrate(ref: Mapping[str, float], dut: Mapping[str, float]) -> str:
    parts: list[str] = []

    # Sharpness: combine the two sharpness metrics into one phrase.
    sh = _pct(dut["sharpness"], ref["sharpness"])
    sh_fft = _pct(dut["sharpness_fft"], ref["sharpness_fft"])
    avg = (sh + sh_fft) / 2
    parts.append(f"sharpness {sh:.1f} / {sh_fft:.1f}% (avg ~{avg:.0f}%)")

    # Noise: lower-better; phrase positively (cleaner) or negatively (noisier).
    n_pct = _pct(dut["noise"], ref["noise"])
    if n_pct < 95:
        parts.append(f"noise {n_pct:.0f}% of ref (cleaner)")
    elif n_pct > 105:
        parts.append(f"noise {n_pct:.0f}% of ref (noisier)")
    else:
        parts.append("noise on par with ref")

    # Channels R/G/B: report the range of % deviation and direction.
    rgb_pct = [
        _pct(dut["R"], ref["R"]),
        _pct(dut["G"], ref["G"]),
        _pct(dut["B"], ref["B"]),
    ]
    deltas = [p - 100 for p in rgb_pct]
    lo, hi = min(deltas), max(deltas)
    if hi < -1.0:
        parts.append(f"channels {abs(hi):.1f}–{abs(lo):.1f}% darker")
    elif lo > 1.0:
        parts.append(f"channels {lo:.1f}–{hi:.1f}% brighter")
    elif lo < -1.0 < hi:
        parts.append(f"channels mixed ({lo:+.1f} to {hi:+.1f}%)")
    else:
        parts.append("channel levels match ref")

    # Color cast: a*/b* absolute shift. Lab a*/b* are signed, so report Δ
    # not ratio (ratio is meaningless when ref ≈ 0).
    da = dut["a*"] - ref["a*"]
    db = dut["b*"] - ref["b*"]
    if abs(da) >= 1.0 or abs(db) >= 1.0:
        parts.append(f"tint shift a*={da:+.2f}, b*={db:+.2f}")
    else:
        parts.append("color stable")

    return "; ".join(parts) + "."
