# Alif ML Embedded Evaluation Kit — C/C++ Standards & Versions

Brief reference of the language standards, toolchains and library versions used
by the Alif fork of the Arm ML Embedded Evaluation Kit.

- **Source reviewed:** `~/projects/rtos_sensor/projects/alif_ml-embedded-evaluation-kit_pg/`
- **Project:** `arm_ml_embedded_evaluation_kit` (PointGrab / Alif fork), commit `964f5a6`
- **Purpose:** bare-metal ML inference on Alif Ensemble (Cortex-M55 + Ethos-U NPU)

## Language standards

| Language | Standard | Enforcement |
| --- | --- | --- |
| C | **C99** (`-std=c99`) | `CMAKE_C_STANDARD_REQUIRED ON` |
| C++ | **C++14** (`-std=c++14`) | `CMAKE_CXX_STANDARD_REQUIRED ON` |

- **Strict ISO**, no GNU extensions — `CMAKE_C_EXTENSIONS OFF` and
  `CMAKE_CXX_EXTENSIONS OFF` (so `c++14`, not `gnu++14`).
- **No C++17** anywhere in the tree.
- The standard is *required*: CMake will not silently fall back to an older one.

## Toolchains & minimum versions

| Tool | Version | Notes |
| --- | --- | --- |
| CMake | **≥ 3.21.0** | `cmake_minimum_required`; presets schema v4 |
| Arm Compiler (armclang) | **≥ 6.19** | Arm Compiler 6; bare-metal target build |
| Arm GNU Toolchain (`arm-none-eabi-gcc`) | **≥ 10.2.1** | pinned path `gcc-arm-none-eabi-10.3-2021.10` |
| Native host GCC | system `gcc`/`g++` | `native-gcc.cmake`, for host/native builds & tests |
| Python | **≥ 3.9** | build scripts and model-zoo tooling |

Both bare-metal compilers are version-checked at configure time and fail hard if
below the minimum (`enforce_compiler_version`).

## Target hardware & CPU flags

- **CPU:** Arm Cortex-**M55** (default) / Cortex-**M85** — Armv8.1-M Mainline.
  (GNU toolchain builds M85 as M55; armclang targets it directly.)
- **SIMD:** Helium / MVE — `-march=armv8.1-m.main+mve.fp+fp.dp`
  (armclang: `--cpu=8.1-M.Main.mve.fp`).
- **Float:** hardware FP, `-mfloat-abi=hard`.
- **NPU:** Arm Ethos-U (microNPU) offload.

## Compile options

**C++ constraints (embedded):**

```
-fno-rtti  -fno-exceptions  -fno-unwind-tables
```

No RTTI and no exceptions — code must not rely on `dynamic_cast`, `typeid`, or
`try`/`throw`/`catch`.

**Optimization per build type:**

| Build type | C / C++ flags |
| --- | --- |
| Debug | `-O1 -g` |
| Release | `-O2 -g -DNDEBUG` (`-O3` on the MVE path) |
| MinSizeRel | `-Os -g -DNDEBUG` |

TensorFlow Lite Micro core and kernels are pinned to `-O0` (known
optimisation-related issue); CMSIS-DSP is forced to `-O3`.

**Warnings** (enabled; **`-Werror` is *not* enforced**):

```
-Wall -Wextra -Wshadow -Wsign-compare -Wswitch
-Wunused-function -Wunused-parameter -Wvla
```

## Key dependency versions

| Component | Version |
| --- | --- |
| CMSIS | **5.9.0** |
| CMSIS-DSP | **1.14.4** |
| CMSIS-NN | **4.0.0** |
| TensorFlow Lite Micro | **1.22.8** |
| RTOS | FreeRTOS and ThreadX (Alif Ensemble) |

CMSIS device support comes from the Alif CMSIS ensemble
(`dependencies/cmsis-ensemble/Alif_CMSIS`).

## Formatting (`.clang-format`)

Based on **LLVM** style, with:

| Setting | Value |
| --- | --- |
| `IndentWidth` | 4 (spaces) |
| `ColumnLimit` | 100 |
| `PointerAlignment` | Left (`int* p`) |
| `AccessModifierOffset` | -4 |
| `NamespaceIndentation` | Inner |
| `BinPackArguments` / `BinPackParameters` | false (one per line when wrapped) |
| `AlignConsecutiveAssignments` / `AlignConsecutiveMacros` | true |
| Braces | `AfterFunction: true` (Allman for functions; attached for classes/control) |
| `BreakConstructorInitializers` / `BreakInheritanceList` | AfterColon |

## Code idioms (observed)

- **File header:** SPDX `Apache-2.0` licence block on every file.
- **Header guards:** `#ifndef NAME_HPP` / `#define` / `#endif` (not `#pragma once`).
- **Namespaces:** nested, e.g. `namespace arm { namespace app { ... } }`.
- **Naming:** `PascalCase` for types *and* methods (`ApplicationContext`,
  `Get()`); member variables prefixed `m_` (`m_value`).
- **C++ features (within C++14):** `override`, `= default`, `explicit`
  constructors, templates, `std::unique_ptr` / `std::shared_ptr` (RAII).
- **Extensions:** headers `.hpp`, sources `.cpp` for C++; `.h`/`.c` for C.

## Quick summary

> **C99 + C++14, strict ISO (no GNU extensions).** Built with Arm Compiler 6
> (armclang ≥ 6.19) or Arm GNU `arm-none-eabi-gcc` ≥ 10.2.1, CMake ≥ 3.21,
> Python ≥ 3.9. Targets Cortex-M55/M85 with Helium/MVE and Ethos-U. **No
> exceptions, no RTTI.** LLVM-based `.clang-format` (4-space indent, 100-col).
> ML stack: CMSIS 5.9.0 / CMSIS-NN 4.0.0 / CMSIS-DSP 1.14.4 / TFLM 1.22.8.

---

# Eigen — Linear Algebra Dependency

Assessment of `~/projects/rtos_sensor/dependencies/eigen` and whether it covers
the matrix operations needed by the pure-NumPy models in
`background_subtraction/scr/models`.

## Language & standards

| | |
| --- | --- |
| Language | C++ — **header-only** template library (include-path only, no build) |
| Version | **Eigen 3.3.9** |
| C++ standard | **C++03 minimum**; uses C++11/14 when available → compatible with the kit's **C++14** and both toolchains |
| License | **MPL-2.0** (a few files BSD/LGPL) |
| Exceptions | Works under `-fno-exceptions`: falls back to `eigen_assert` when `EIGEN_EXCEPTIONS` is undefined — matches the kit's `-fno-exceptions -fno-rtti` |
| Modules present | Core, Dense, LU, QR, **SVD (JacobiSVD, BDCSVD)**, Cholesky, Eigenvalues, Geometry, Sparse, plus the *unsupported/experimental* N-D **Tensor** module |

## How the kit already uses Eigen

`source/application/api/use_case/pg_object_detection/src/kalmanFilter.cc` (the
ByteTrack Kalman filter) uses Eigen the canonical way — **small, fixed-size
dense matrices** (`dataType.hpp`: `Eigen::Matrix<float, 8, 8>`,
`<float, 1, 8>`, …) with `Identity()`, matrix `*`, `.transpose()`,
`.asDiagonal()`, `.block<>()`, `.array().square()` (element-wise) and
`Eigen/Cholesky` (LLT solve). So Eigen is a **vetted dependency already
compiling under the embedded toolchain**, but the established pattern is *small
dense linear algebra*, not image-sized tensor processing.

## Coverage for `scr/models`

The models are written in **NumPy N-D tensor style** (H×W×C, H×W×K×C,
broadcasting, per-axis indexing). Eigen's **stable API is 2-D** dense
matrices/arrays.

| Model | Operations it needs | Eigen (stable) coverage |
| --- | --- | --- |
| **EigenBackground** | mean, **SVD** (PCA), matrix multiply | ✅ **Full** — `BDCSVD`/`JacobiSVD` + `*` + `colwise().mean()`; the one model that genuinely needs matrix ops |
| FrameDifference | abs, sub, max-over-channel, compare | ⚠️ Expressible via `Array` (`.abs()`, `.cwiseMax`, `.select()`), but trivial pixel math |
| RunningAverage | `α·x+(1-α)·bg`, abs, select | ⚠️ `Array` + `.select()` |
| AdaptiveMedian | sign, abs, any-over-channel, conditional | ⚠️ `Array` + `.sign()` + `.select()` |
| RunningGaussian | squared-dist (sum over C), variance update, clip | ⚠️ element-wise + reductions |
| **MixtureOfGaussians** | N-D tensor, **argmax/argsort along an axis**, gather (`take_along_axis`), `cumsum` along axis, per-pixel mode replacement | ❌ **Not** in stable Eigen — needs explicit loops (or the unsupported Tensor module) |
| **ViBe** | sample buffer (H×W×N), random scatter, neighbour diffusion | ❌ **Not** linear algebra — stateful random indexing |

## Bottom line

- Eigen **fully covers EigenBackground** (SVD + matmul — precisely its purpose).
- It *can* express the four element-wise models via its `Array` type, but they
  need no matrix library at all (plain loops or CMSIS-DSP/Helium do it, often
  preferable on Cortex-M).
- It **does not cover the tensor-indexing core of MixtureOfGaussians and ViBe**
  (axis-wise argsort/gather/cumsum; random sample-buffer scatter) — those stay
  hand-written per-pixel loops on the MCU regardless. The experimental N-D
  `Tensor` module exists in the tree but is unsupported, template-heavy, and
  unused by the kit.
- **Port guidance:** use Eigen the narrow way the Kalman filter does — for the
  small linear-algebra kernel (EigenBackground's SVD/matmul). Implement
  FrameDiff / RunningAverage / Median / Gaussian as element-wise loops
  (optionally CMSIS-DSP), and MOG / ViBe as explicit per-pixel loops over small
  fixed K/N structures.
