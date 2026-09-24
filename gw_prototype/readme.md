# gw_prototype

Python prototype of the PG2 gateway: receives detections from Alif sensors over MQTT, filters
them by detection area, converts pixel coordinates to centimetres through the fisheye model,
stores them in SQLite and aggregates counts for areas of interest and XY events. Ships sensor
and cloud stubs so the whole chain runs on one machine.

**Status:** ARCHIVED — frozen 2025-09. Superseded by `PG2_GW`, but still the fastest local test rig.

## Purpose

Prove the gateway's data path end to end before committing it to C: sensor → MQTT → geometry →
storage → aggregation → cloud. Because it ships `sensor_stub.py` and `cloud_stub.py` alongside
a mosquitto broker in Docker, the entire pipeline runs without hardware — which is why it
remains useful even though production moved on.

The module layout mirrors the responsibilities:

| Module | Role |
|---|---|
| `gw_entry_point.py` | Main loop |
| `mqtt_client.py`, `proto/` | Transport: mosquitto + protobuf serialisation |
| `preprocessor.py`, `preprocessing_utils.py` | Filtering and coordinate conversion |
| `fisheye_camera.py`, `geometry.py`, `detection_area.py` | The camera model and area logic |
| `xy_storage.py` | SQLite persistence |
| `aggregator.py`, `aoi.py` | AoI counts and XY events |
| `sensor_stub.py`, `cloud_stub.py` | Emulated endpoints |
| `tests/`, `perf/` | pytest suite and performance probes |

## Connections

- **Superseded by `PG2_GW`** (the shipping C gateway) and by `gw_cpp_version` (the C++
  intermediate step, which reuses *this* project's broker and stubs).
- `fisheye_camera.py` / `geometry.py` / `detection_area.py` were **ported to C++ as
  `pg_camera_geometry`**. Together with `PG2_GW/src` that makes **four** implementations of the
  same geometry; the shipping one is in `PG2_GW`.
- Consumes lens parameters produced by `camera_calibration/`; imported directly by
  `kpmg_research/` and by `camera_calibration`'s visualisation notebooks.
- `algo_system_emulator/src/gw/` carries its geometry and AoI logic.
- Holds `camera_calibration/calib-cam4revb/` — a byte-identical 344 MB copy of the captures in
  `camera_calibration/data/`.

## Dependencies

`reqirements.txt` (sic): `protobuf==3.20.3`, `paho-mqtt==2.1.0`, `numpy==2.1.3`,
`pytest==8.3.3`, `pytest-mock==3.14.0`. Plus Docker for mosquitto and `libprotoc` to regenerate
the protobuf stubs.

---

## Reqirements

* Linux OS
* Python 3.10.9
* Docker
* libprotoc 3.12.4

## Description

The prototype implements main functionality of the PG2 Gateway:
* receiving detections from Alif sensors
* preprocessing: filtration by detection area, conversion from px to cm
* storing to SQLite database
* aggregation: count for AoI and XY events

The communication between Sensor, GW and Cloud implemented using mosquitto MQTT broker and protobuf serialization.

Also, the prototype has sensor and cloud stubs.

## How to launch

1. Mosquitto (v. 2.0.20):
```bash
cd mosquitto
docker compose up
```

2. Cloud stub:
```bash
python cloud_stub.py
```

3. Gateway
```bash
python gw_entry_point.py
```

4. Sensor stub
```bash
python sensor_stub.py
```

## Raspbrry Pi setup
1. Copy source code to device (updat the destination before)
```bash
bash sync.sh
```
Next steps on the device

2. Install pip and virtualenv if needed
```bash
sudo apt install -y python3-pip python3-virtualenv
```

3. Setup GW python env
```bash
cd gw && python3 -m virtualenv .venv
source .venv/bin/activate
pip install -r reqirements.txt 
```

4. Install and setup mosquitto (https://randomnerdtutorials.com/how-to-install-mosquitto-broker-on-raspberry-pi/)
```bash
cd mosquitto
bash setup_mosquitto.sh
```
