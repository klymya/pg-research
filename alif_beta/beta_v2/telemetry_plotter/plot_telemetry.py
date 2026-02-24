#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle


class Detection:
    """Represents a single detection"""

    def __init__(
        self,
        pixel_x,
        pixel_y,
        cm_x,
        cm_y,
        conf,
        box_x,
        box_y,
        box_w,
        box_h,
        sensor_id,
        label="translated",
        gmm_label=None,
    ):
        self.pixel_x = pixel_x
        self.pixel_y = pixel_y
        self.cm_x = cm_x
        self.cm_y = cm_y
        self.confidence = conf
        self.box_x = box_x
        self.box_y = box_y
        self.box_w = box_w
        self.box_h = box_h
        self.sensor_id = sensor_id
        self.label = label
        self.gmm_label = gmm_label


class TelemetryMessage:
    """Represents a telemetry message with detections"""

    def __init__(self, timestamp, sensor_id, length):
        self.timestamp = timestamp
        self.sensor_id = sensor_id
        self.length = length
        self.detections = []

    def add_detection(self, detection):
        self.detections.append(detection)

    def plot(
        self,
        ax=None,
        colors=None,
        show_pixels=True,
        show_conf=True,
        title=None,
        image_path: Optional[Path] = None,
    ):
        """Plot detections in pixel coordinates with top-left origin."""
        if colors is None:
            colors = {
                "translated": "tab:blue",
                "gmm_filtered": "tab:red",
            }

        if ax is None:
            _, ax = plt.subplots()

        # Fixed 320x240 canvas, origin at top-left
        ax.set_xlim(0, 320)
        ax.set_ylim(240, 0)
        ax.set_aspect("equal", adjustable="box")

        if image_path is not None:
            img = mpimg.imread(image_path)
            ax.imshow(img, extent=(0, 320, 240, 0))

        for det in self.detections:
            color = colors.get(det.label, "gray")
            rect = Rectangle(
                (det.box_x, det.box_y),
                det.box_w,
                det.box_h,
                fill=False,
                linewidth=2,
                edgecolor=color,
            )
            ax.add_patch(rect)

            if show_pixels and det.pixel_x is not None and det.pixel_y is not None:
                ax.scatter([det.pixel_x], [det.pixel_y], s=20, color=color)

            if show_conf:
                ax.text(
                    det.box_x + det.box_w,
                    det.box_y,
                    f"{det.confidence:.2f}",
                    ha="right",
                    va="top",
                    color=color,
                    fontsize=8,
                    bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1),
                )

        ax.set_title(title or f"{self.sensor_id} @ {self.timestamp}")
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")
        return ax


def parse_log_file(log_file_path: Path) -> List[TelemetryMessage]:
    """
    Parse the log file and extract telemetry messages with their detections.
    """
    telemetry_pattern = re.compile(
        r"(\w{3}\s+\d{2}\s+\d{2}:\d{2}:\d{2}).*Received telemetry data, length: (\d+), from thing ([\w\-]+)"
    )

    detection_pattern = re.compile(
        r"(\w{3}\s+\d{2}\s+\d{2}:\d{2}:\d{2}).*Detection translated for sensor ([\w\-]+): "
        r"pixel\((\d+),(\d+)\) -> cm\((-?\d+),(-?\d+)\). "
        r"conf = ([\d.]+); box: x = (\d+), y = (\d+), w = (\d+), h = (\d+)"
    )

    gmm_filtered_pattern = re.compile(
        r"(\w{3}\s+\d{2}\s+\d{2}:\d{2}:\d{2}).*Detection filtered out by GMM: (\w+) "
        r"\(x=(\d+), y=(\d+), w=(\d+), h=(\d+), conf=([\d.]+)\)"
    )

    telemetry_messages = []
    current_message = None

    with open(log_file_path, "r") as f:
        for line in f:
            telemetry_match = telemetry_pattern.search(line)
            if telemetry_match:
                if current_message is not None:
                    telemetry_messages.append(current_message)

                timestamp = telemetry_match.group(1)
                length = int(telemetry_match.group(2))
                sensor_id = telemetry_match.group(3)
                current_message = TelemetryMessage(timestamp, sensor_id, length)
                continue

            detection_match = detection_pattern.search(line)
            if detection_match and current_message is not None:
                sensor_id = detection_match.group(2)
                pixel_x = int(detection_match.group(3))
                pixel_y = int(detection_match.group(4))
                cm_x = int(detection_match.group(5))
                cm_y = int(detection_match.group(6))
                conf = float(detection_match.group(7))
                box_x = int(detection_match.group(8))
                box_y = int(detection_match.group(9))
                box_w = int(detection_match.group(10))
                box_h = int(detection_match.group(11))

                detection = Detection(
                    pixel_x,
                    pixel_y,
                    cm_x,
                    cm_y,
                    conf,
                    box_x,
                    box_y,
                    box_w,
                    box_h,
                    sensor_id,
                    label="translated",
                )
                current_message.add_detection(detection)
                continue

            gmm_match = gmm_filtered_pattern.search(line)
            if gmm_match and current_message is not None:
                gmm_label = gmm_match.group(2)
                box_x = int(gmm_match.group(3))
                box_y = int(gmm_match.group(4))
                box_w = int(gmm_match.group(5))
                box_h = int(gmm_match.group(6))
                conf = float(gmm_match.group(7))

                detection = Detection(
                    None,
                    None,
                    None,
                    None,
                    conf,
                    box_x,
                    box_y,
                    box_w,
                    box_h,
                    current_message.sensor_id,
                    label="gmm_filtered",
                    gmm_label=gmm_label,
                )
                current_message.add_detection(detection)

    if current_message is not None:
        telemetry_messages.append(current_message)

    return telemetry_messages


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot telemetry detections from a log file."
    )
    parser.add_argument(
        "log_file",
        type=Path,
        help="Path to the pg2gw log file",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Optional background image path",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Telemetry message index to plot (default: 0)",
    )
    args = parser.parse_args()

    if not args.log_file.exists():
        raise SystemExit(f"Log file not found: {args.log_file}")

    if args.image is not None and not args.image.exists():
        raise SystemExit(f"Image file not found: {args.image}")

    messages = parse_log_file(args.log_file)
    if not messages:
        raise SystemExit("No telemetry messages found in log file")

    if args.index < 0 or args.index >= len(messages):
        raise SystemExit(
            f"Index out of range: {args.index}. Available: 0..{len(messages)-1}"
        )

    msg = messages[args.index]
    msg.plot(image_path=args.image)

    output_path = Path(__file__).parent / "output.png"
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
