import subprocess
import sys
import os
import argparse


def run_cmd(command: list) -> None:
    print(f"Running command: {' '.join(command)}")
    try:
        _ = subprocess.run(
            command,
            stdout=sys.stdout,  # Stream stdout to console
            stderr=sys.stderr,  # Stream stderr to console
            check=True,         # Raise CalledProcessError on failure
        )
        print(f"Successfully processed command: {' '.join(command)}")
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {' '.join(command)}")
        print(e.stderr)
        sys.exit(e.returncode)


def run_yolo(dataset, cfg, weights, exp_name, device, batch):
    command = [
        "python", "train_yolo.py",
        "--dataset", dataset,
        "--cfg", cfg,
        "--weights", weights,
        "--exp_name", exp_name,
        "--device", ' '.join([str(d) for d in device]),
        "--batch", str(batch),
    ]

    run_cmd(command)


def run_export(weights, data, max_ncalib_imgs):
    command = [
        "python", "export.py",
        "--weights", weights,
        "--data", data,
        "--max_ncalib_imgs", str(max_ncalib_imgs),
        "--imgsz", "320",
        "--quantize"
    ]

    run_cmd(command)


def run_vela_command(weights_path):
    command = [
        "python", "optimize_vela.py", weights_path,
    ]

    run_cmd(command)


def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Run vela commands for best_saved_model and head_saved_model.")
    parser.add_argument("--dataset", type=str, default="baseline")
    parser.add_argument("--exp_name", type=str, default="baseline")
    parser.add_argument("--cfg", type=str, default="yolov8-pose-xs.yaml")
    parser.add_argument("--weights", type=str, default="yolov8s-pose.pt")
    parser.add_argument("--device", nargs="+", type=int, default=[0])
    parser.add_argument("--batch", type=int, default=32)

    parser.add_argument("--max_ncalib_imgs", type=int, default="100", help="calibration image set size")

    args = parser.parse_args()

    run_yolo(
        args.dataset,
        args.cfg,
        args.weights,
        args.exp_name,
        args.device,
        args.batch
    )

    run_export(
        os.path.join(os.getcwd(), "results", args.exp_name, "weights", "best.pt"),
        os.path.join(args.dataset, "project.yaml"),
        args.max_ncalib_imgs
    )

    run_vela_command(os.path.join(os.getcwd(), "results", args.exp_name, "weights"))

if __name__ == "__main__":
    main()
