from datetime import datetime
import subprocess
import argparse
import os
import glob


if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))

    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--weights",
        type=str,
        default="../people_detection_pg2/results/filtered_v3_margin_15/weights/best.pt",
        help="Path to the YOLOv8 model file."
    )
    argparser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Confidence threshold for detection."
    )
    argparser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="IoU threshold for NMS."
    )
    argparser.add_argument(
        "--device",
        type=int,
        default=1,
        help="Device ID for GPU processing."
    )
    argparser.add_argument(
        "--output_root_dir",
        type=str,
        default=os.path.join(dir_path, f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
        help="Output directory for processed files."
    )
    argparser.add_argument(
        "--pgpytrain_path",
        type=str,
        default=os.path.join(dir_path, "../../PG_PyTraining"),
        help="Output directory for processed files."
    )
    args = argparser.parse_args()

    subprocess.run([
        "python",
        os.path.join(dir_path, "eval.py"),
        "--weights", args.weights,
        "--conf", str(args.conf),
        "--iou", str(args.iou),
        "--device", str(args.device),
        "--output_root_dir", args.output_root_dir,
    ], check=False)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{env.get('PYTHONPATH', '')}:{args.pgpytrain_path}"

    result_subfolder_paths = glob.glob(os.path.join(args.output_root_dir, "*"))
    os.chdir(args.pgpytrain_path)
    for result_subfolder_path in result_subfolder_paths:
        subprocess.run([
            "python",
            "benchmarking/bma.py",
            "--i", result_subfolder_path,
        ], env=env, check=False)
    os.chdir(dir_path)

    subprocess.run([
        "python",
        os.path.join(dir_path, "compute_avg_score.py"),
        "-i", args.output_root_dir,
    ], check=False)
