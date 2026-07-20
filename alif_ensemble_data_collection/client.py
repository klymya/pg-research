import serial
import time
import os
from datetime import datetime

# Get metadata from user
camera_height = input("Enter camera height (e.g. 2.8m): ").replace(" ", "_")
camera_type = input("Enter camera type (e.g. RGB): ").replace(" ", "_")
FILE_AMOUNT = int(input("Enter number of images to capture: "))

# Check and print working directory
print("Current working directory:", os.getcwd())

# Configure the serial port (Change as needed)
SERIAL_PORT = "/dev/ttyACM0"
BAUD_RATE = 115200
FILE_SIZE = 921776  # 640*480*3 + 176 tiff header size
file_index = 0

# Create date-based parent directory (e.g. "2025-04-29")
shooting_date = datetime.now().strftime("%Y-%m-%d")
date_dir = os.path.join(os.getcwd(), shooting_date)
print("Creating date directory:", date_dir)

try:
    os.makedirs(date_dir, exist_ok=True)
    print(f"Date directory created or already exists: {date_dir}")
except Exception as e:
    print(f"Failed to create date directory: {e}")

# Create time-based session directory inside date directory
session_time = datetime.now().strftime("%H-%M-%S")
session_dir_name = f"4.2_H{camera_height}_{session_time}"
session_dir = os.path.join(date_dir, session_dir_name)
print("Creating session directory:", session_dir)

try:
    os.makedirs(session_dir, exist_ok=True)
    print(f"Session directory created: {session_dir}")
except Exception as e:
    print(f"Failed to create session directory: {e}")

# Create file prefix using metadata
OUTPUT_FILE_PREFIX = f"4.2_H_cam_{camera_height}_{camera_type}_"
print("File prefix will be:", OUTPUT_FILE_PREFIX)

while FILE_AMOUNT > 0:
    FILE_AMOUNT -= 1
    file_size_left = FILE_SIZE

    try:
        file_index += 1
        formatted_file_index = f"{file_index:04d}"

        temporary_file_name = os.path.join(
            session_dir,
            OUTPUT_FILE_PREFIX + formatted_file_index + ".tmp"
        )

        with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) as ser, \
                open(temporary_file_name, "wb") as file:

            print(f"Recording data from {SERIAL_PORT}...")

            ser.write(b'\x05')

            # Timestamp when reading from serial starts, in milliseconds.
            read_start_timestamp_ms = time.time_ns() // 1_000_000
            read_start_monotonic = time.monotonic()

            while file_size_left:
                data = ser.read(file_size_left)
                print("Received bytes:", len(data))
                file_size_left -= len(data)

                if data:
                    file.write(data)
                    file.flush()
                else:
                    break

            # Duration of serial reading, in milliseconds.
            read_duration_ms = round(
                (time.monotonic() - read_start_monotonic) * 1000
            )

        FILE_NAME = os.path.join(
            session_dir,
            (
                f"{OUTPUT_FILE_PREFIX}"
                f"{formatted_file_index}_"
                f"{read_start_timestamp_ms}_"
                f"{read_duration_ms}.tif"
            )
        )

        os.rename(temporary_file_name, FILE_NAME)
        print("Saved image:", FILE_NAME)

    except serial.SerialException as e:
        print(f"Serial error: {e}")
    except KeyboardInterrupt:
        print("\nRecording stopped by user.")
