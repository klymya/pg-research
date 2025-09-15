import subprocess
import sys
import os
import argparse


def run_vela_command(weights_path, output_dir):
    command = [
        "vela", weights_path,
        "--output-dir", output_dir,
        "--accelerator-config", "ethos-u55-128",
        "--optimise", "Performance",
        "--config", "vela.ini",
        "--system-config", "Ethos_U55_High_End_Embedded",
        "--memory-mode", "Shared_Sram",
        "--show-cpu-operations",
    ]

    print(f"Running command: {' '.join(command)}")
    try:
        _ = subprocess.run(
            command,
            stdout=sys.stdout,  # Stream stdout to console
            stderr=sys.stderr,  # Stream stderr to console
            check=True           # Raise CalledProcessError on failure
        )
        print(f"Successfully processed {weights_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error running vela command for {weights_path}:")
        print(e.stderr)
        sys.exit(e.returncode)


def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Run vela commands for best_saved_model and head_saved_model.")
    parser.add_argument("weights_path", type=str, help="Path to the directory containing the weights and models.")
    args = parser.parse_args()

    weights_path = args.weights_path

    # Run vela command for best_saved_model
    best_model_path = os.path.join(weights_path, "best_saved_model", "best_full_integer_quant.tflite")
    best_output_dir = os.path.join(weights_path, "best_saved_model", "vela_he")
    run_vela_command(best_model_path, best_output_dir)

    # Run vela command for head_saved_model
    head_model_path = os.path.join(weights_path, "head_saved_model", "head_float32.tflite")
    head_output_dir = os.path.join(weights_path, "head_saved_model", "vela_he")
    run_vela_command(head_model_path, head_output_dir)

if __name__ == "__main__":
    main()
