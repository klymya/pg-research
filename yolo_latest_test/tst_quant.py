import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from ultralytics import YOLO

# m = YOLO('yolo11n.pt')
# success = m.export(format='tflite', int8=True, data='coco128.yaml', imgsz=640, nms=False)
# print(success)
YOLO('pg2/best.pt').export(format='tflite', imgsz=320, nms=False, int8=True, data='/home/klymy/data/cam4_revB/yolo_dataset/batch1_5_margin_15_bg_wb_alif_clahe/project.yaml')
