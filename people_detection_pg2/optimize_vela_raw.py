import argparse
import os

from optimize_vela import run_vela_command


args = argparse.ArgumentParser(description="Run vela commands for tflite weights.")
args.add_argument("--weights-path", type=str, help="Path to the directory containing the weights and models.")
args.add_argument("--output-dir", type=str, default="", help="Output directory for the optimized models.")
args = args.parse_args()

if not args.output_dir:
    args.output_dir = os.path.join(os.path.dirname(args.weights_path), "vela_he")

run_vela_command(
    args.weights_path,
    args.output_dir
)
