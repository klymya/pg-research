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
2. Record video and timestamps on the device using `capture.py` and record the radar data using `vayyar.py` 
3. Put the resulting data together
4. Run `render_vayyar_data.py` to create video with points projection. Please add path to PG_PyTraining repo to `PYTHONPATH`, because the script requires tools from it.