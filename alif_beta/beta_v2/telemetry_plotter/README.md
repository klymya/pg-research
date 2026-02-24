# Telemetry Plotter

Plots translated and GMM-filtered detections from a pg2gw log file. The output image is saved as `output.png` in the same folder as the script.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python plot_telemetry.py example.log
```

Optional background image and index:

```bash
python plot_telemetry.py example.log --image example.jpg --index 0
```

## Notes

- Canvas is fixed to 320x240 with top-left origin.
- Translated detections are blue, GMM-filtered detections are red.
- Confidence is drawn at the top-right corner of each box.
