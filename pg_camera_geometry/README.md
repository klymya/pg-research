# pg_camera_geometry

C++ camera-geometry library: fisheye projection, pixel↔centimetre conversion and
detection-area membership. The C++ port of `gw_prototype`'s Python geometry.

**Status:** ARCHIVED — frozen 2024-12.

## Purpose

Turn a detection's pixel coordinates into a position on the floor, and decide whether that
position lies inside the configured detection area — in C++, so the gateway does not need
Python.

Three translation units, each mirroring a Python module in `gw_prototype`:

| Header / source | Mirrors | Responsibility |
|---|---|---|
| `FisheyeCamera.hpp/.cpp` | `fisheye_camera.py` | Lens model, undistortion, px↔cm |
| `Geometry.hpp/.cpp` | `geometry.py` | Primitives — points, polygons, transforms |
| `DetectionArea.hpp/.cpp` | `detection_area.py` | Detection-area polygon and membership tests |

`test/` holds a gtest suite per unit (`TestFisheyeCamera`, `TestGeometry`,
`TestDetectionArea`), which is the clearest specification of the intended behaviour.

## Connections

- **Port of `gw_prototype`'s geometry**, consumed conceptually by `gw_cpp_version` and
  `PG2_GW`.
- **This is one of four implementations of the same geometry**: here, `gw_prototype` (Python),
  `PG2_GW/src` (shipping C) and `PG_PyTraining/common/pg/pg_geometry.py`. There is no single
  source of truth, and they have drifted. **The shipping implementation is in `PG2_GW`** —
  treat this one as reference, not authority.
- Consumes lens parameters produced by `camera_calibration/`.

## Dependencies

vcpkg (`vcpkg.json`): `gtest` only — the library itself is dependency-free.
Build: CMake with presets (`CMakePresets.json`), vcpkg toolchain.

The 204 MB on disk is a local `vcpkg/` checkout and `build/` output; both are gitignored, and
only 17 files are tracked.
