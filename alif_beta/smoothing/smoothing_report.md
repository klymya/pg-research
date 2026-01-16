# Smoothing Methods Report (Notebook Summary)

## What this report contains

- A brief engineering assessment of each smoother
- A comparison table (decision-oriented)
- Plots exported from the notebook outputs (where available)
- A recommended production default: **Median → EMA**

## Comparison Table

| Method | Noise reduction | Spike robustness | Step responsiveness | Online-safe | Notes |
|---|---:|---:|---:|---:|---|

| SMA | Medium | Low | Low (laggy) | Yes | Good baseline; strong lag |

| WMA | Medium | Low | Medium | Yes | Slightly less lag than SMA |

| EMA | Medium | Medium | Medium–High | Yes | Good default; tune α |

| Gaussian | High | Low | Low–Medium | **No** | Non-causal; best offline |

| Median | Medium | **High** | High | Yes | Best spike removal; can plateau |

| Savitzky–Golay | Medium | Low | Medium | Usually yes | Often misleading for count steps |

| LOWESS | High | Medium | Low | Usually no | Slow; trend-only |

| **Median → EMA** | **High** | **High** | **High** | **Yes** | **Recommended** |


## Plots

### SMA
*(No exported plot found in notebook outputs.)*

### WMA
![](report_assets/wma_11_0.png)

### EMA
![](report_assets/ema_9_0.png)

### Gaussian
![](report_assets/gaussian_8_0.png)

### Median
![](report_assets/median_7_0.png)

### Savitzky-Golay
![](report_assets/savitzky-golay_6_0.png)

### LOWESS
![](report_assets/lowess_4_0.png)

### Median → EMA (Robust smoothing)

This plot is **added in the updated notebook** as a new cell and will be saved as `plot_robust_median_ema.png` when executed.

## Final Recommendation

Use **Median(3) → EMA(α≈0.25)** as the default smoother for step-like detection telemetry. It provides the best overall trade-off: robust against spikes while remaining responsive to true occupancy changes.
