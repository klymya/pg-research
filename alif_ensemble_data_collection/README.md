# alif_ensemble_data_collection

Alif Ensemble firmware for **capturing still frames off the sensor and shipping them to a host**,
plus the host-side receiver. This is the tool that produces new in-house image datasets.

**Status:** ACTIVE · last change 2026-09 · 478 MB

## Purpose

Every in-house dataset starts as frames captured through the real lens, on the real sensor, in
the real installation. This firmware does that: the HM0360 captures a RAW8 Bayer snapshot,
applies automatic gain analysis, converts Bayer→RGB888, prepends a 176-byte TIFF header and
hands back a complete in-memory TIFF.

**Frame format:** 640 × 480 × 3 + 176-byte header = **921,776 bytes**. That is exactly the size
of every `.tif` in `camera_calibration/data/` — those calibration captures came from here.

`client.py` is the host side: prompts for camera height and type, opens `/dev/ttyACM0` at
115200 baud, and writes each fixed-size payload to a named `.tif`.

> **Known limitation, documented in `CAMERA_TIFF_TRANSFER.md` (review dated 2026-07-20):** the
> checked-in `usb` build **does not currently transfer the image**. `WITH_GOTO_STOP` is
> enabled, so the app captures one snapshot then enters the low-power flow; and all USB init
> and TIFF transmission are guarded by `DC_BOARD`, which is absent from `usb.cproject.yml` and
> the generated build files. The UART JPEG path exists but its writes are `#if 0`. So the
> repository holds an *intended* USB transfer path, while the current build only captures and
> builds the TIFF in memory. Read that document before trying to collect data.

## Layout

| Path | What |
|---|---|
| `CAMERA_TIFF_TRANSFER.md` | **Read this first** — end-to-end review of the capture and transfer path, with a source-file map |
| `client.py` | Host-side receiver over USB CDC-ACM |
| `analyze_transfer.ipynb` | Transfer analysis |
| `board/image/source/ensemble/` | Capture pipeline: `camera.c`, `tiff.c`, `bayer2rgb.c`, `image_ensemble.c` |
| `usb/`, `ethernet/`, `sd_card/`, `blinky/` | Transport variants and the upstream examples |
| `use_case/pg_object_detection*` | The on-device ML use case (where a deployed model runs) |
| `data_example/` | Sample captures |

## Connections

- **Produces the raw captures** consumed by `camera_calibration/` (the 921,776-byte TIFFs) and
  by the dataset builds in `people_detection_pg2/cam4_revB/`.
- `use_case/pg_object_detection/` is the on-device counterpart to the model
  `people_detection_pg2` trains and `cp_weights.py` ships into `rtos_sensor`.
- Same Alif Ensemble target as `background_subtraction`'s Sign-of-Life C port and
  `alif_perf`'s cycle-count parsing.

## Dependencies

**Firmware:** CMSIS-Toolbox, Arm GNU toolchain, CMake, Ninja, Alif SE tools, J-Link — managed
through Arm Environment Manager (`vcpkg-configuration.json`). See the upstream notes below.

**Host:** Python with `pyserial`.

```bash
pip install pyserial
python client.py
```

---

## Upstream template notes

> Everything below is the **original Alif VS Code template README**, kept for its toolchain and
> environment setup instructions. It describes the upstream `blinky` / `ethernet` / `usb` /
> `sd_card` examples, **not** PointGrab's capture tool. Forked from
> `alifsemi/alif_ensemble-vscode-azure-blinky`.

# VSCode Azure RTOS examples
A simple Azure RTOS and CMSIS-Pack based example to setup and test VS Code development environment for Alif Ensemble kits.

- Arm GNU toolchain is used as a default. There are build-type options for IAR and ARM compiler armclang for reference.
  - You can find the compiler specific settings in `cdefault.yaml`
  - **TIP:** The tools loaded by Arm Environment Manager are configured in `vcpkg-configuration.json`.
  - To download armclang you can add "arm:compilers/arm/armclang": "^6.22.0" to the "requires" object.

# Projects in this repository
- blinky
  - This example demonstrates GPIO usage using a button press to generate and GPIO interrupt which toggles the LED state.
- ethernet
  - Simple DHCP example
  - printf is retargeted to UART4
- usb
  - Acts as a serial device. Echoes input to output
  - printf is retargeted to UART4
- sd_card
  - A readily FAT32 formatted SD card is expected. Partition should be smaller than 32GIB
  - printf is retargeted to UART4

## Note about Ensemble gen1 support
This example supports only Gen2 Ensemble Devices

## Quick start
First clone the project repository
```
git clone https://github.com/alifsemi/alif_ensemble-vscode-azure-blinky.git
```

To build the project for a supported board other than the DevKit, you have to update the `board.h` file to pick the right variant of the board.
By default the template will build for gen2 DevKit.

The required software setup consists of *VS Code*, *Git*, *CMake*, *Ninja build system*, *cmsis-toolbox*, *Arm GNU toolchain* and *Alif SE tools*.
By default the template project uses J-link so *J-link software* is required for debugging.
In addition to build tools the VS Code extensions and CMSIS packs will be downloaded automatically during the process.

To make environment setup easier this project uses *Arm Environment Manager* for downloading and configuring most of the tool dependencies.
Basically only VS Code, Alif SE tools and J-Link software need to be downloaded and installed manually.

Opening the project folder with VS Code automatically suggests installing the extensions needed by this project:
- Arm Environment Manager
- Arm CMSIS csolution
- Cortex-Debug
- Microsoft C/C++ Extension Pack

After setting up the environment you can just click the CMSIS icon and then the *context* and *build* icon to get going.

For Alif SE tools and J-link debugging support add the following entries to VS Code user settings.json (Press F1 and start typing 'User')
```
{
    "alif.setools.root" : "C:/alif-se-tools/app-release-exec",
    "cortex-debug.JLinkGDBServerPath": "C:/Program Files/SEGGER/JLink/JLinkGDBServerCL.exe"
}
```

## More detailed getting started guide
Please refer to the template project's [Getting started guide](https://github.com/alifsemi/alif_vscode-template/blob/master/doc/getting_started.md)

## Examples for Alif Peripherals
More Azure Alif peripheral examples can be found from https://github.com/alifsemi/alif_ensemble-Azure-RTOS/tree/main/THREADX/samples
