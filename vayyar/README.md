# vayyar

Evaluation of the Vayyar 60 GHz imaging radar as a people-sensing alternative, including
projecting its returns onto simultaneously recorded video.

**Status:** ARCHIVED — frozen 2023-11 · 111 MB

## Purpose

Radar detects presence and movement without capturing an image, which matters where cameras are
unwelcome. The question was whether its point returns are good enough to count and locate
people.

The interesting part is the **validation method**: rather than judging radar output in
isolation, the radar and a PG camera record the same scene simultaneously, and the radar points
are projected onto the video frame. A human can then see directly whether a return corresponds
to a person.

| Script | Role |
|---|---|
| `src/vayyar.py` | Record radar data from the Vayyar engine over its WebSocket API |
| `src/capture.py` | Record video and timestamps on the PG device |
| `src/render_vayyar_data.py` | Project radar points onto the video — the main output |
| `src/draw_vayyar_data.py` | Overlay points without projection |
| `src/vayyr_data_to_seq.py` | Build a `.seq` file and timestamp-aligned video |
| `src/point_mapper.py`, `src/sort/` | Point mapping and SORT tracking |
| `src/metric_test_cases.py` | Test cases for the metrics |

`previous-research/` holds the earlier iteration; `Documentation/` the vendor material.

## Outcome

Not adopted; the camera pipeline continued. No written conclusion — the rendered videos are the
record.

## Connections

- Parallel to `Socionext/`, the other 60 GHz radar evaluated around the same time.
- **Requires `PG_PyTraining`** on `PYTHONPATH` — `render_vayyar_data.py` uses its tooling, and
  `pg_pytraining/` here is a partial copy.
- Produces `.seq` files of the kind `data-exploration/` indexes.

## Dependencies

`requirements.txt` declares only `numpy` and `websocket-client`. The scripts additionally
import `opencv-python`, `matplotlib`, `tqdm`, `scipy`, `scikit-learn`, `filterpy` and `lap`
(tracking) — **undeclared**, so a fresh install from the requirements file will not run them.

Needs the **Vayyar_IndoorEVK_vTrig_v7.1.0** engine installed on a Windows PC with the radar on
USB — the capture path cannot run without that hardware.

---

## Original notes

- previous-research - scripts from previouse research iteration.
- src/vayyar.py - script for the radar data writing.
- Vayyar_IndoorEVK_vTrig_v7.1.0_setup.exevayyar/Documentation - Vayyar engine.
- Documentation - Vayyar radar documentation.

## Usage:

1. Install Vayyar_IndoorEVK_vTrig_v7.1.0 on a Windows PC.
2. Install Python and pip.
3. Install reqirements
```bash
pip install -r requirements.txt
```
4. Connect the radar using USB.
5. Run the Vayyar engine.
6. Run the data writing script
```bash
py src/vayyar.py
```


## Data collection
1. Setup the radar and the PG device. Write installation parameters to `params.json`
2. Record video and timestamps on the device using `src/capture.py` and record the radar data using `src/vayyar.py` 
3. Put the resulting data together
4. Run `src/render_vayyar_data.py` to create video with points projection. Please add path to PG_PyTraining repo to `PYTHONPATH`, because the script requires tools from it.

## Additional scripts
- src/draw_vayyar_data.py - draw vayyar points withoud projection on the PG video.
- src/vayyr_data_to_seq.py - create seq file and aligned by timestamps video from the vayyar data file and the original PG videofile.