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


def run_yolo(dataset, cfg, weights, exp_name, device, batch, yolo_kwargs):
    command = [
        "python", "train_yolo.py",
        "--dataset", dataset,
        "--weights", weights,
        "--exp_name", exp_name,
        "--device", ' '.join([str(d) for d in device]),
        "--batch", str(batch),
        "--kwargs", yolo_kwargs,
    ]

    if cfg != "":
        command += ["--cfg", cfg]

    run_cmd(command)


def run_export(weights, data, max_ncalib_imgs, separated_head=False, output_folder_name="best_saved_model"):
    if separated_head:
        command = [
            "python", "export.py",
            "--weights", weights,
            "--data", data,
            "--max_ncalib_imgs", str(max_ncalib_imgs),
            "--imgsz", "320",
            "--quantize"
        ]
    else:
        command = [
            "python", "export_non_separated.py",
            "--weights", weights,
            "--data", data,
            "--max_ncalib_imgs", str(max_ncalib_imgs),
            "--imgsz", "320",
            "--output-folder-name", output_folder_name
        ]

    run_cmd(command)


def run_vela_command(weights_path, best_saved_model_name=None, skip_head=False):
    command = [
        "python", "optimize_vela.py", weights_path, 
    ]

    if best_saved_model_name is not None:
        command.append(best_saved_model_name)
    if skip_head:
        command.append("--skip-head")

    run_cmd(command)


def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Run vela commands for best_saved_model and head_saved_model.")
    parser.add_argument("--dataset", type=str, default="baseline")
    parser.add_argument("--exp_name", type=str, default="baseline")
    parser.add_argument("--cfg", type=str, default="")
    parser.add_argument("--weights", type=str, default="yolov8s-pose.pt")
    parser.add_argument("--device", nargs="+", type=int, default=[0])
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--yolo_kwargs", type=str, default="")

    parser.add_argument("--max_ncalib_imgs", type=int, default="100", help="calibration image set size")
    parser.add_argument(
        "--converted-output-folder-name",
        type=str, default="best_saved_model", help="Output folder name for converted model"
    )

    args = parser.parse_args()

    run_yolo(
        args.dataset,
        args.cfg,
        args.weights,
        args.exp_name,
        args.device,
        args.batch,
        args.yolo_kwargs,
    )

    # export separated head
    run_export(
        os.path.join(os.getcwd(), "results", args.exp_name, "weights", "best.pt"),
        os.path.join(args.dataset, "project.yaml"),
        args.max_ncalib_imgs,
        separated_head=True,
    )
    run_vela_command(os.path.join(os.getcwd(), "results", args.exp_name, "weights"), "best_saved_model")

    # export non-separated head
    run_export(
        os.path.join(os.getcwd(), "results", args.exp_name, "weights", "best.pt"),
        os.path.join(args.dataset, "project.yaml"),
        args.max_ncalib_imgs,
        separated_head=False,
        output_folder_name=args.converted_output_folder_name
    )
    run_vela_command(
        os.path.join(os.getcwd(), "results", args.exp_name, "weights"),
        best_saved_model_name=args.converted_output_folder_name,
        skip_head=True,
    )

if __name__ == "__main__":
    main()
