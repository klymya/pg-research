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
