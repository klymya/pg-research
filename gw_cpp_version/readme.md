# gw_cpp_version

The PG2 gateway reimplemented in C++: the same receive → filter → convert → store → aggregate
chain as `gw_prototype`, built with CMake and vcpkg. Reuses `gw_prototype`'s mosquitto broker
and its sensor and cloud stubs for testing.

**Status:** ARCHIVED — frozen 2025-01. Superseded by `PG2_GW`.

## Purpose

The intermediate step between the Python prototype and the shipping gateway: establish that the
data path survives translation to C++ with acceptable performance, before committing to the
production implementation.

| Component | Mirrors |
|---|---|
| `MQTTClientHandler` | `mqtt_client.py` |
| `PreProcessor`, `PreProcessingUtils` | `preprocessor.py`, `preprocessing_utils.py` |
| `XYStorage` | `xy_storage.py` (SQLite) |
| `Aggregator`, `AreaOfInterest` | `aggregator.py`, `aoi.py` |

`test/` carries a gtest suite covering each of those.

> **The heading below is wrong.** The preserved section still reads "Gateway **Python**
> Prototype" — it was copy-pasted from `gw_prototype/readme.md` and never updated. The build
> and launch instructions in it are correct for this project; only the title was stale.

## Connections

- **Superseded by `PG2_GW`.**
- Depends on `gw_prototype` at runtime for the mosquitto compose file and the stubs — this
  project has no emulators of its own.
- Geometry lives separately in `pg_camera_geometry`; see the four-copy note in
  `gw_prototype/readme.md`.

## Dependencies

vcpkg (`vcpkg.json`): `gtest`, `sqlite3`, `paho-mqtt`, `protobuf`, `inih`.
Toolchain: CMake 3.31.2, ninja 1.10.1, `libprotoc` 3.12.4.

---

## Reqirements

* Linux OS
* ninja 1.10.1
* cmake 3.31.2
* Python 3.10.9
* Docker
* libprotoc 3.12.4
* vscode + vcpkg


## Description

The prototype implements main functionality of the PG2 Gateway using C++:
* receiving detections from Alif sensors
* preprocessing: filtration by detection area, conversion from px to cm
* storing to SQLite database
* aggregation: count for AoI and XY events

The communication between Sensor, GW and Cloud implemented using mosquitto MQTT broker and protobuf serialization.

The sensor and cloud emulators, mqtt broker are in separate folder `../gw_prototype`.

## How to build

Use vcpkg to install dependencies and vscode with CMake extension to build the project.

## How to launch

1. Mosquitto:
```bash
cd ../gw_prototype/mosquitto
docker compose up
```

2. Cloud stub:
```bash
cd ../gw_prototype
python cloud_stub.py
```

3. Gateway
```bash
./build/pg_gateway
```

4. Sensor stub
```bash
cd ../gw_prototype
python sensor_stub.py
```