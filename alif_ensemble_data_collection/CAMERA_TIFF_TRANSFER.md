# Camera Capture and TIFF Transfer Review

Review date: 2026-07-20

## Executive summary

The camera path captures an HM0360 RAW8 Bayer snapshot into a static buffer, performs automatic gain analysis, converts the Bayer frame to RGB888, prepends a 176-byte TIFF header, and returns a pointer to the complete in-memory TIFF image.

The firmware does **not** write a file to an on-device filesystem. The intended USB path sends the in-memory TIFF bytes over USB CDC-ACM; software on the receiving device must read exactly one complete payload and save it as a `.tif` or `.tiff` file.

The checked-in `usb` build does not currently transfer the image:

- `WITH_GOTO_STOP` is enabled, so the active application captures one snapshot and then enters the low-power flow.
- All USB initialization and TIFF transmission are guarded by `DC_BOARD`, but `DC_BOARD` is absent from `usb.cproject.yml`, the generated `usb.*.cbuild.yml` files, and the recorded compile commands.
- The UART JPEG sender contains encoding logic, but its actual UART writes are disabled with `#if 0`. There is no active TIFF-over-UART implementation.

Therefore, the repository contains an intended USB TIFF transfer path, but the current build configuration only captures and constructs the TIFF in memory.

## Relevant source files

| Responsibility | Source |
| --- | --- |
| USB application and transfer loop | [`usb/demo_usbx_device_cdc_acm_app.c`](usb/demo_usbx_device_cdc_acm_app.c) |
| Active USB project source list | [`usb/usb.cproject.yml`](usb/usb.cproject.yml) |
| Image buffers and capture pipeline | [`board/image/source/ensemble/image_ensemble.c`](board/image/source/ensemble/image_ensemble.c) |
| Camera controller wrapper | [`board/image/source/ensemble/src/camera.c`](board/image/source/ensemble/src/camera.c) |
| TIFF header generation | [`board/image/source/ensemble/src/tiff.c`](board/image/source/ensemble/src/tiff.c) |
| TIFF structure definition | [`board/image/source/ensemble/include/tiff.h`](board/image/source/ensemble/include/tiff.h) |
| Bayer-to-RGB conversion | [`board/image/source/ensemble/src/bayer2rgb.c`](board/image/source/ensemble/src/bayer2rgb.c) |
| Image dimensions and pixel sizes | [`board/image/source/ensemble/include/image_processing.h`](board/image/source/ensemble/include/image_processing.h) |
| HM0360 configuration | [`device/RTE/Device/AE302F80F55D5LE_M55_HE/RTE_Device.h`](device/RTE/Device/AE302F80F55D5LE_M55_HE/RTE_Device.h) |
| Inactive JPEG/UART path | [`board/image/source/ensemble/src/jpeg.c`](board/image/source/ensemble/src/jpeg.c) |
| General UART transport | [`hal/ensemble/source/uart_tracelib.c`](hal/ensemble/source/uart_tracelib.c) |

The similarly named `usb/my_demo_usbx_device_cdc_acm_app.c` and `usb/demo_usbx_device_cdc_acm_app_2025-05-12.c` are not listed as application sources in `usb/usb.cproject.yml` and are not part of the active path reviewed here.

## End-to-end logic

1. `main()` initializes board pinmux, system services, clocks, and the ThreadX kernel.
2. `tx_application_define()` creates the application thread. USBX initialization only occurs when `DC_BOARD` is defined.
3. `demo_thread_entry()` calls `image_init()`.
4. `image_init()` passes the static `raw_image` buffer to `camera_init()`.
5. `camera_init()` initializes and powers the camera controller, configures the sensor, registers capture/error events, and retains the raw buffer address.
6. `get_image_data()` starts `CAMERA_MODE_SNAPSHOT` and waits up to 100 ms for capture completion.
7. The camera controller/DMA writes one byte per sensor pixel into `raw_image`; the CPU invalidates its data cache for that buffer after capture.
8. The code analyzes the RAW8 frame and adjusts sensor gain for a later capture.
9. `write_tiff_header()` writes TIFF metadata at the beginning of `rgb_image`.
10. `dc1394_bayer_Simple()` demosaics the RAW8 Bayer data into interleaved RGB888 immediately after the header.
11. `get_image_data()` returns a pointer to the start of `rgb_image`, not merely to its pixel array. The pointer therefore addresses a complete TIFF byte stream.
12. In the `DC_BOARD` branch, the host must first send at least one byte. The firmware then calls `ux_device_class_cdc_acm_write()` with a hard-coded length of 921,776 bytes.
13. The receiving device is responsible for accumulating the entire payload and writing it to a file.

There is no overlap or double buffering: capture, gain analysis, demosaicing, and transfer are performed serially using one raw buffer and one TIFF/RGB buffer.

## Image and TIFF format

### Sensor input

| Property | Configured value |
| --- | --- |
| Sensor | HM0360 |
| Sensor transport | MIPI CSI-2, one lane |
| CSI-2 data type | `42` / `0x2A`, RAW8 |
| Configured dimensions | 640 × 480 from HM0360 context A |
| Raw storage | 8-bit Bayer, 307,200 bytes |
| Configured sensor rate | 60 FPS |

### TIFF output

| Property | Value |
| --- | --- |
| Dimensions | 640 × 480 |
| Photometric interpretation | RGB |
| Pixel layout | Interleaved RGB888, 3 bytes per pixel |
| Bits per sample | 8, 8, 8 |
| Compression | None |
| Strip layout | One strip containing all 480 rows |
| Byte order | Little-endian (`II`) |
| TIFF magic | 42 |
| Header size in the current GCC ABI | 176 bytes |
| Pixel data size | 640 × 480 × 3 = 921,600 bytes |
| Total transmitted size | 176 + 921,600 = **921,776 bytes** |
| Approximate size | 900.2 KiB / 0.879 MiB |

The TIFF buffer layout is:

```text
offset 0                                      offset 176
+---------------------------------------------+--------------------------------+
| TIFF header and one Image File Directory    | RGB888 pixels                  |
| 176 bytes                                   | 921,600 bytes                  |
+---------------------------------------------+--------------------------------+
                                              total: 921,776 bytes
```

No JPEG or other compression is applied to the USB payload. Scaling and white-balance calls are present but commented out, so the function arguments named `ml_width` and `ml_height` do not change the generated dimensions.

## Frame rate and throughput

The configured sensor rate is not the delivered transfer rate.

| Stage or mode | Rate |
| --- | --- |
| Sensor configuration | 60 FPS, or one sensor period every 16.7 ms |
| Snapshot wait | Up to 100 ms before capture is treated as failed |
| Active checked-in build | One capture followed by stop-mode entry; it is not a video stream |
| `WITH_GOTO_STOP` plus intended `DC_BOARD` flow | One host-triggered frame, 2-second delay, then a 5-second low-power timer; at most about 0.14 frame/s before initialization, processing, transfer, and host wait are included |
| Alternative continuous branch | No measured or enforced FPS; each iteration waits for capture, processes the frame, waits for a host request, and completes the USB write |

At 60 FPS, uncompressed TIFF payloads alone require approximately 55.3 MB/s (52.8 MiB/s). This is close to the 60 MB/s raw signaling ceiling of USB 2.0 High Speed and leaves insufficient allowance for USB protocol overhead, CDC-ACM overhead, software copies, capture, or demosaicing. Sustained 60 FPS TIFF transfer should not be expected from this design.

The USB descriptors request 512-byte bulk endpoints, which indicates an intended USB High Speed connection. Actual FPS must be measured on hardware because the code has no frame counter, timestamps, profiling output, or transport benchmark.

For comparison only, transmitting the same 921,776-byte uncompressed TIFF over an 8-N-1 UART would take approximately:

| UART rate | Ideal transfer time | Ideal maximum rate |
| --- | ---: | ---: |
| 921,600 baud | 10.0 s | 0.10 frame/s |
| 115,200 baud | 80.0 s | 0.0125 frame/s |

These are wire-time lower bounds and exclude processing and framing. The 921,600-baud rate is configured by `platform_drivers.c` for the trace UART in another application path; the USB project itself does not enable TIFF transmission through that UART.

## Transfer protocols

### USB CDC-ACM

The intended protocol is host-triggered and has no explicit application-level framing:

1. The host opens the CDC-ACM device and sends any non-empty request of up to 512 bytes.
2. The device sends exactly 921,776 raw bytes beginning with the TIFF `II` signature.
3. The host must continue reading until all 921,776 bytes have been accumulated.
4. The host saves those bytes unchanged as a TIFF file.

There is no command identifier, payload-length field, frame sequence number, checksum, acknowledgement, retry, or end marker. The receiving program is not present in this repository.

### UART

There is no active TIFF-over-UART path.

`jpeg_send_through_uart()` encodes RGB data as a quality-85 JPEG and defines an intended framing format consisting of marker bytes `AB BA`, a native `size_t` length, and chunks of up to 256 bytes. However, all `send_str()` calls in the active C implementation are inside `#if 0`, and the USB application's call to this function is also compiled out. This code currently only encodes a JPEG and prints debugger instructions.

The general UART helper `send_str()` is blocking: it starts a CMSIS USART transfer and waits indefinitely for the send-complete event. If restored for images, a stalled UART can therefore block the entire image pipeline.

## Restrictions and limitations

### Critical functional issues

1. **USB is disabled by the current build configuration.** `DC_BOARD` guards USB clock setup, USBX initialization, device callbacks, and the image write. It is not defined by the active project or generated build configuration. The recorded release binary also has a very small `demo_thread_entry`, consistent with the USB transfer branch being removed by preprocessing.

2. **The camera resolution configuration conflicts with the buffer dimensions.** The generated component macro is `RTE_Drivers_CPI`, but `camera.c` tests `RTE_CPI`. Because `RTE_CPI` is not defined, the code selects `Driver_CAMERA0` and explicitly requests `CAMERA_RESOLUTION_560x560`. The raw buffer is only 640 × 480 = 307,200 bytes, while a 560 × 560 RAW8 frame requires 313,600 bytes. If the driver honors the 560 × 560 request, capture can overrun `raw_image` by 6,400 bytes and the later 640 × 480 interpretation is incorrect.

3. **The Bayer pattern selection is inconsistent.** `BAYER_FORMAT` selects BGGR for silicon revision A and GRBG otherwise, but the real conversion call always passes `DC1394_COLOR_FILTER_BGGR`. Non-revision-A hardware may therefore produce swapped or otherwise incorrect colors.

4. **The USB transfer size is hard-coded.** `921776` depends on 640 × 480 RGB888 and the compiler-specific 176-byte `tiff_header_t` layout. Changing resolution, packing, compiler ABI, or the TIFF structure can cause truncation or an out-of-bounds read. There is no compile-time assertion tying the constant to `sizeof(rgb_image)`.

### Reliability limitations

- The active one-shot path checks a failed capture and then spins forever. The alternative loop does not check whether `get_image_data()` returned `NULL` before attempting transfer.
- `camera_start()` ignores the return value from `CaptureFrame()`.
- `camera_wait()` does not check the return status from `tx_event_flags_get()` separately.
- If both capture-complete and error bits are set, `camera_wait()` accepts the frame because it only rejects an error when capture-complete is absent.
- USB code does not wait explicitly for `g_cdc` to become non-null before its first read.
- `ux_device_class_cdc_acm_write()` status is checked, but the returned `actual_length` is not verified against 921,776 and a partial transfer is not retried.
- There is no payload integrity check or way for the receiver to detect corruption other than parsing the TIFF.
- There is no timeout on the host request in the USB path, so a missing host can block the application indefinitely.
- The TIFF/RGB buffer is not explicitly cleaned from the CPU data cache before USB transfer. Whether this is safe depends on whether USBX copies through coherent/non-cacheable memory or the device controller accesses the application buffer directly.

### Format and processing limitations

- Output is always full-resolution 640 × 480 despite the dimensions passed to `get_image_data()`; resize code is disabled.
- White balance and color correction are disabled.
- Demosaicing adds a black border on the last row and last column.
- The automatic-gain update affects a subsequent capture, not the frame currently returned.
- The TIFF uses a single uncompressed strip. It requires the complete 921,776-byte buffer in memory and offers no bandwidth reduction.
- The TIFF structure is serialized by copying a native C structure. It relies on little-endian execution and the current compiler's padding/alignment rather than explicitly writing portable byte offsets.
- The USB full-speed descriptor also advertises 512-byte bulk packets; USB full-speed bulk endpoints normally require a smaller maximum packet size. Full-speed fallback should be validated or corrected.

### Memory footprint

The capture/TIFF path reserves:

| Buffer | Size |
| --- | ---: |
| RAW8 capture | 307,200 bytes |
| TIFF header plus RGB888 | 921,776 bytes |
| Capture/TIFF subtotal | **1,228,976 bytes** |
| JPEG scratch buffer, if the inactive JPEG path is linked | 307,200 bytes |
| Capture/TIFF plus JPEG | **1,536,176 bytes** |

The recorded release ELF contains `raw_image` and `rgb_image`; linker garbage collection removes the unused JPEG buffer in that build. These figures exclude USBX memory (32 KiB when enabled), the application thread stack (32 KiB), RTOS objects, application data, and driver/DMA storage.

## Practical conclusions

- Treat the current implementation as an in-memory snapshot producer, not as a working TIFF file-transfer feature.
- The intended USB payload is a valid-looking uncompressed 640 × 480 RGB TIFF of 921,776 bytes, but the host-side receiver is missing and the active build excludes the USB branch.
- UART image transfer is only an unfinished JPEG design and cannot currently transfer TIFF or JPEG bytes.
- Before hardware use, resolve the 560 × 560 versus 640 × 480 capture mismatch; it is a potential memory-corruption issue.
- A robust transfer implementation should derive payload size from the buffer, add explicit framing and integrity metadata, handle partial I/O and disconnects, and measure end-to-end FPS on the target hardware.
