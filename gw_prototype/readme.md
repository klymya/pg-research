# Gateway Python Prototype

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

1. Mosquitto:
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