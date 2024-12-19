# Gateway Python Prototype

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