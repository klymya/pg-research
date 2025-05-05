import argparse
import os
import shutil


PATHS = [
    (
        "weights/best_saved_model/best_full_integer_quant.tflite",
        "backbone.tflite",
    ),
    (
        "weights/best_saved_model/vela_he/best_full_integer_quant_vela.tflite",
        "backbone_H128.tflite",
    ),
    (
        "weights/best_saved_model/vela_he/best_full_integer_quant_summary_Ethos_U55_High_End_Embedded.csv",
        "backbone_summary.csv",
    ),
    (
        "weights/head_saved_model/head_float32.tflite",
        "head.tflite",
    ),
    (
        "weights/head_saved_model/vela_he/head_float32_vela.tflite",
        "head_H128.tflite",
    ),
    (
        "weights/head_saved_model/vela_he/head_float32_summary_Ethos_U55_High_End_Embedded.csv",
        "head_summary.csv",
    ),
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy weights from one directory to another.")
    parser.add_argument(
        "--src",
        type=str,
        required=True,
        help="Path to the source root directory containing the weights.",
    )
    parser.add_argument(
        "--dst",
        type=str,
        required=True,
        help="Path to the destination directory where the weights will be copied.",
    )
    args = parser.parse_args()

    # Create destination directory if it doesn't exist
    os.makedirs(args.dst, exist_ok=True)

    # Copy all files from source to destination
    for base_src, base_dest in PATHS:
        src_path = os.path.join(args.src, base_src)
        dest_path = os.path.join(args.dst, base_dest)
        if os.path.exists(src_path):
            shutil.copy(src_path, dest_path)
            print(f"Copied {base_src} to {base_dest}")
        else:
            print(f"Source file {base_src} does not exist.")
