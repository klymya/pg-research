# Sign-of-Life pipeline — C++14 reference

A C++14 implementation of the "Sign of Life" background-subtraction pipeline, a
faithful port of the pure-NumPy Python reference in
[`background_subtraction/scr`](../scr). It reads a folder of JPEG frames and, per
frame, produces the foreground mask, the mask after morphology, and a per-frame
**Sign-of-Life** flag with the list of foreground objects.

It is written to the **Alif ML embedded evaluation kit** C++ standards (C++14,
strict ISO, no exceptions/RTTI in the algorithm code, `arm`-style formatting) so
the algorithm sources can be lifted onto the Cortex-M55/M85 target. See
[`../native_operators.md`](../native_operators.md) for the full porting notes.

## Pipeline

Each frame flows through:

1. **Read JPEG → RGB** (`ImageIo`, stb_image).
2. **Gaussian blur** denoise (`NativeOps::GaussianBlur`).
3. **Grayscale feature + running-Gaussian background model** → raw foreground
   (`NativeOps::RgbToGray`, `BackgroundModel`).
4. **HSV shadow filtration** — removes cast-shadow pixels
   (`NativeOps::RgbToHsv`, `ShadowSuppressor`).
5. **Morphology** (elliptical open → close) on the foreground mask
   (`NativeOps::MorphologyEx`).
6. **Sign-of-Life** — connected components + area gate; the frame is flagged
   when any surviving region's area ≥ `--sol-area` (`NativeOps::ConnectedComponents`,
   `Pipeline`).

The pipeline is **stateful** (the model and shadow reference persist across
frames), so frames are processed in file-name order.

## Layout

```
sign_of_life_cpp/
├── CMakeLists.txt
├── README.md
├── src/
│   ├── Types.hpp            # Image, Blob, FrameResult, PipelineConfig
│   ├── NativeOps.{hpp,cpp}  # RGB→gray/HSV, Gaussian, morphology, conn-components
│   ├── BackgroundModel.*    # running Gaussian average (Wren)
│   ├── ShadowSuppressor.*   # HSV veto + EW RGB reference
│   ├── Pipeline.*           # stage orchestration + Sign-of-Life
│   ├── ImageIo.*            # host: JPEG decode / PNG write / dir listing
│   └── Main.cpp             # CLI driver
└── third_party/
    ├── stb_image.h          # JPEG/PNG decode (public domain, v2.30)
    └── stb_image_write.h    # PNG encode (public domain, v1.16)
```

**No external dependencies** — image I/O uses the vendored public-domain stb
single-header libraries; directory listing uses POSIX `dirent` (host only).

## Build

Requires CMake ≥ 3.21 and a C++14 compiler (GCC or Clang).

```bash
cd background_subtraction/sign_of_life_cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

The executable is `build/sign_of_life`.

## Run

```
sign_of_life <input_dir> <output_dir> [--min-area N] [--sol-area N] [--no-shadow-veto]
```

| Argument | Meaning |
| --- | --- |
| `<input_dir>` | Folder of JPEG/PNG frames, processed in name order |
| `<output_dir>` | Folder for the output masks and report (created if missing) |
| `--min-area N` | Smallest kept foreground object, px (default `150`) |
| `--sol-area N` | Sign-of-Life area threshold, px (default `500`) |
| `--no-shadow-veto` | Disable the HSV shadow filter (default: enabled) |

`--no-shadow-veto` is the same gate as `main.py --no-shadow-veto` and the Alif
port's `PipelineConfig::shadowEnabled`. When it is off the veto's background
reference — a full-frame float RGB buffer, 921,600 B at QVGA — is never
allocated, and the HSV conversions are never run. The setting is recorded in the
`sign_of_life.txt` header as `shadow_veto=0/1`.

Example:

```bash
./build/sign_of_life ../data_0_1fps/busStation/input out_busStation
```

## Outputs

Written to `<output_dir>`:

- **`<frame>_raw.png`** — the raw foreground mask (model output after the shadow
  veto), single-channel 0/255.
- **`<frame>_morph.png`** — the mask after morphology and the area gate (only
  surviving objects).
- **`sign_of_life.txt`** — per-frame report:

  ```text
  # Sign-of-Life report
  # min_area=150 area_threshold=500
  # columns: frame  sign_of_life  num_objects  objects[area@x,y,w,h;...]
  in000001.jpg  0  0
  in000481.jpg  1  1  2909@55,70,51,94
  in000961.jpg  1  4  2560@52,54,66,101;2098@128,67,36,104;157@49,139,18,23;151@77,157,28,13
  # summary: 6 frames, 3 sign-of-life
  ```

  Each object is `area@x,y,w,h` (bounding box), largest first.

## Parity with the Python reference

- **Integer ops are bit-exact** with the Python/`cv2` reference: `RgbToGray`
  (BT.601), `RgbToHsv` (fixed-point), the elliptical structuring element,
  erode/dilate/open/close, and connected-components stats.
- **Float ops match within rounding**: the running-Gaussian model (single-
  precision) and the HSV ratio test. `GaussianBlur` is a float separable
  Gaussian with the cv2 sigma formula and reflect-101 border — very close to,
  but not bit-identical with, `cv2.GaussianBlur`.
- **Resolution:** this tool processes each JPEG at its **native size** (no
  resize). The Python notebook resizes to QVGA 320×240, so object *areas* scale
  with resolution — for already-QVGA inputs (e.g. `canoe`) the results are
  near-identical; for larger inputs (e.g. `busStation` at 360×240) the areas are
  proportionally larger. The Sign-of-Life flag pattern matches either way.

## Notes for the Alif port

The algorithm translation units (`NativeOps`, `BackgroundModel`,
`ShadowSuppressor`, `Pipeline`, `Types`) use only C++14 + the standard library
and are ready to compile under the eval-kit toolchain. Only `ImageIo` and
`Main` are host-specific (stb + `dirent` + CLI); on device, frames come from the
camera HAL (already **RGB, QVGA** — see `../native_operators.md`), so those two
files are replaced by the kit's PreProcess/PostProcess wiring.
