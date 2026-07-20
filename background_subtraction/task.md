# PG2: [Algo] Prototype 4-Layer Non-NN Background Subtraction Pipeline in Python

## Description

Develop a modular Python prototype to research and evaluate the confirmed 4-layer background subtraction (BGS) architecture for "Sign of Life" detection. The system must process video files using OpenCV and the bgslibrary wrapper to isolate newly introduced static objects while suppressing dynamic office noise (shadows, lighting shifts, and camera drift).

The prototype must implement the following unified architectural flow:

```txt
[ Frame Input ]
       │
       ▼
 ┌───────────┐
 │  Layer 0  │ Preprocessing (Denoise, Color Space, Registration)
 └─────┬─────┘
       │  (Cleaned Frame)
       ▼
 ┌───────────┐     ┌───────────┐
 │  Layer 2  │ ──> │  Layer 1  │ Illumination Invariance -> Rate-Parameterized BGS Component
 └───────────┘     └─────┬─────┘
                         │  (Instantiated Twice)
                         ▼
                   ┌───────────┐
                   │  Layer 3  │ Dual-Rate BGS (Fast + Slow) + FSM & Dwell-Time Engine
                   └─────┬─────┘
                         │
                         ▼
               [ "Sign of Life" Signal ]
```

Layer 0: Preprocessing

Handle camera sensor noise, frame-to-frame variance, and minor hardware drift.

Implementation: Apply Gaussian or Bilateral filtering, convert frames to target color-spaces (Grayscale/HSV), and explore basic global frame registration/stabilization if needed.

Layer 1 & 2: Rate-Parameterized Model & Illumination Invariance

Layer 1 must be built as a highly reusable, parameterized class where the learning rate ($\alpha$) is an input argument.

Integrate Layer 2 directly: use texture metrics (Local Binary Patterns - LBP) or chromaticity variations instead of raw pixel intensities as the differencing metric. Implement an HSV brightness-ratio threshold as a post-filter to veto foreground noise caused by moving shadows.

Core Algorithms to Evaluate (via OpenCV / bgslibrary):

Median/Running-Average (Primary baseline matching 1–6 frame initialization).

ViBe / SuBSENSE (Available in bgslibrary for tracking multimodal backgrounds).

MOG2/KNN (For local evaluation baseline).

Layer 3: Dual-Rate Framework & Finite State Machine (FSM)

Instantiate two parallel copies of your Layer 1 module:

Fast Background Model ($\alpha_{fast}$): Absorbs changes quickly to track transient shadows/motion.

Slow Background Model ($\alpha_{slow}$): Absorbs changes slowly to preserve newly static objects.

Build an FSM / Dwell-time comparison engine: if an item is present in the Fast mask but absent in the Slow mask for a specified dwell-time threshold (without intersecting an active human tracking bounding box), trigger the "Sign of Life" left-object state.

Incorporate basic structural filters here (blob-area gating to ignore tiny pixel clusters).

Research Assets & Technical Requirements

Language: Python 3.x

Core Packages: opencv-python

Extended BGS Suite: Use the Python wrapper for andrewssobral/bgslibrary to easily swap out underlying algorithms.

Test Datasets: Download and run the evaluation scripts using the benchmark videos provided in the [CDNET-14](https://changedetection.net/dataset2014/)
