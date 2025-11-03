import argparse
import os
import shutil


PATHS = [
    (
        "weights/{folder_name}/best_full_integer_quant.tflite",
        "backbone.tflite",
    ),
    (
        "weights/{folder_name}/vela_he/best_full_integer_quant_vela.tflite",
        "backbone_H128.tflite",
    ),
    (
        "weights/{folder_name}/vela_he/best_full_integer_quant_summary_Ethos_U55_High_End_Embedded.csv",
        "backbone_summary.csv",
    ),
]

HEAD_PATHS = [
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
    parser.add_argument(
        "--folder-name",
        type=str,
        required=False,
        default="best_saved_model",
        help="Name of the folder containing the weights to be copied.",
    )
    parser.add_argument(
        "--ignore-head",
    required=False,
        default=False,
        action='store_true',
        help="If set, the head weights will not be copied.",
    )
    args = parser.parse_args()

    # Create destination directory if it doesn't exist
    os.makedirs(args.dst, exist_ok=True)

    paths = [(i[0].format(folder_name=args.folder_name), i[1]) for i in PATHS]
    if not args.ignore_head:
        paths += HEAD_PATHS

    # Copy all files from source to destination
    for base_src, base_dest in paths:
        src_path = os.path.join(args.src, base_src)
        dest_path = os.path.join(args.dst, base_dest)
        if os.path.exists(src_path):
            shutil.copy(src_path, dest_path)
            print(f"Copied {base_src} to {base_dest}")
        else:
            print(f"Source file {base_src} does not exist.")

    with open(os.path.join(args.dst, "source_paths.txt"), "w") as f:
        for base_src, base_dest in paths:
            f.write(f"{base_src}\n")
