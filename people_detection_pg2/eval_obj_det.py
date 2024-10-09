import glob
import os
import subprocess

import numpy as np
import cv2
from tqdm import tqdm

from tflite_inference import TFLiteInferer

import sys
sys.path.append("../objectdetectionmetrics/")
from metricsevaluator import MetricsEvaluator


CONF_TH = 0.35
IOU_TH = 0.5
W = 320

def process_prediction(predictions):
    labels = predictions.boxes.cls.cpu().numpy()[predictions.boxes.cls.cpu().numpy() < 2]
    confidences = predictions.boxes.conf.cpu().numpy()[predictions.boxes.cls.cpu().numpy() < 2]
    boxes = predictions.boxes.xywhn.cpu().numpy()[predictions.boxes.cls.cpu().numpy() < 2]
    
    return os.path.basename(predictions.path).split(".")[0], labels, confidences, boxes


def process_yolo(model, img_pth):
    return process_prediction(model.predict(img_pth, conf=CONF_TH, iou=IOU_TH, verbose=False)[0])


def process_tflite(inferer, img_pth):
    img = cv2.cvtColor(cv2.imread(img_pth), cv2.COLOR_BGR2RGB)
    
    ORIGINAL_H, ORIGIAL_W, _ = img.shape
    RATIO = W / ORIGIAL_W

    detections = inferer.infer(img)

    if detections.shape[0]  == 0:
        boxes = []
        labels = []
        confidences = []
    else:
        boxes = detections[:, :4] / RATIO
        boxes /= [ORIGIAL_W, ORIGINAL_H, ORIGIAL_W, ORIGINAL_H]
        boxes[:,:2] += boxes[:, 2:] / 2
        labels = detections[:, -3]

        confidences = detections[:, -4]

    return img_pth.split("/")[-1].split(".")[0], labels, confidences, boxes


def eval(model, postprocessor, imgs, labels):
    predictions = []
    for img in tqdm(imgs, desc="model inference"):
        predictions.append(postprocessor(model, img))

    # init evaluator
    evaluator = MetricsEvaluator(
        is_relative_coordinates=True,
        is_xywh_coordinates=True,
        img_size=(W, W),
        outpath="."
    )

    for label_pth in labels:
        img_name = label_pth.split("/")[-1].split(".")[0]

        with open(label_pth, "rt") as f:
            lines = f.readlines()
        
        for line in lines:
            c, x_c, y_c, w, h = [float(i) for i in line[:-1].split()][:5]
            if c > 1:
                continue
            if np.isnan(x_c):
                continue
            
            evaluator.add_gt_box(
                img_path=img_name,
                class_name="person",
                bbox=[x_c, y_c, w, h]
            )
        # break

    for img_predictions in predictions:
        img_name, pred_labels, confidences, boxes = img_predictions
        for box, cl, conf in zip(boxes, pred_labels, confidences):
            evaluator.add_det_box(
                img_path=img_name,
                class_name="person",
                bbox=box,
                confidence=conf
            )

    results = evaluator.eval_conf_vals(0.5)

    for result in results:
        precision = result["total TP"] / (result["total TP"] + result["total FP"])
        recall = result["total TP"] / (result["total positives"])

        f_score = 2 * precision * recall / (precision + recall)
        print(f"{result['class']}: P = {precision:.03f}. R = {recall:.03f}. F1 = {f_score:.03f}. AP = {result['AP']:.03f}")


algotest_imgs = sorted(glob.glob("/home/klymy/data/people_detection/filtered_v3_margin_15/images/algotest/*.jpg"))
algotest_labels = sorted(glob.glob("/home/klymy/data/people_detection/filtered_v3_margin_15/labels/algotest/*.txt"))

holdout_imgs = sorted(glob.glob("/home/klymy/data/people_detection/filtered_v3_margin_15/images/holdout/*.jpg"))
holdout_labels = sorted(glob.glob("/home/klymy/data/people_detection/filtered_v3_margin_15/labels/holdout/*.txt"))

for batch_size in [300, 500, 700, 900]:
    subprocess.run([
        "python", "export.py", 
        "--weights", "results/filtered_v3_margin_15/weights/best.pt",
        "--data", "/home/klymy/data/people_detection/filtered_v3_margin_15/project_holdout.yaml",
        "--max_ncalib_imgs", str(batch_size),
        "--imgsz", str(W),
        "--quantize",
    ])

    inferer = TFLiteInferer(
        yolo_pth="results/filtered_v3_margin_15/weights/best_saved_model/best_full_integer_quant.tflite",
        head_pth="results/filtered_v3_margin_15/weights/head_saved_model/head_float32.tflite",
        conf_th=CONF_TH,
        iou_th=IOU_TH,
        imgsz=W,
    )

    print("\t\t", batch_size)
    print("\tALGOTEST:")
    eval(inferer, process_tflite, algotest_imgs, algotest_labels)

    print("\tHOLDOUT:")
    eval(inferer, process_tflite, holdout_imgs, holdout_labels)
    print("===========================================================\n")
