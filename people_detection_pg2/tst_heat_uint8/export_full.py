import argparse
import os
import sys
import glob

import numpy as np
import torch
import cv2

from ultralytics import YOLO


model = YOLO("/home/klymy/projects/researches/people_detection_pg2/tst_heat_uint8/best_full.pt")

model.export(
    format="tflite",
    simplify=True,
    imgsz=320,
    data="/home/klymy/data/cam4_revB/yolo_dataset/batch1_5_margin_15_bg_wb_alif_clahe/project.yaml",
    int8=True,
    export_hw_optimized=True,
    uint8_io_dtype=True,
    max_ncalib_imgs=100,
    separate_outputs=False,
)
