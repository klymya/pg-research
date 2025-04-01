import os
import argparse

import numpy as np

import quality_reduce_comparison.preprocessing as preprocessing
from optimize_vela import run_vela_command


argparser = argparse.ArgumentParser(description="Export input adatper.")
argparser.add_argument(
    "--output", type=str, default="input_adapter"
)
argparser.add_argument(
    "--dataset", type=str
)

args = argparser.parse_args()



y = np.zeros((1, 3, 240, 320), dtype=np.float32)

net = preprocessing.InputAdapter()

preprocessing.model2tflite(
    args.output,
    net,
    y,
    args.dataset,
    150,
)


run_vela_command(
    os.path.join(args.output, "input_adapter_full_integer_quant.tflite"),
    os.path.join(args.output, "vela_he"),
)