from typing import Sequence, Optional, Tuple
import os

import numpy as np

from BoundingBox import BoundingBox
from BoundingBoxes import BoundingBoxes
from utils import BBFormat, BBType, CoordinatesType, MethodAveragePrecision
from Evaluator import Evaluator


def array2str(x):
    return np.array2string(x, formatter={'float': lambda x: f'{x:.2f}'}, separator=', ', max_line_width=1e9)


def get_scores_by_conf_th(scores, confs, conf_ths=None):
    if conf_ths is None:
        conf_ths = np.append(np.arange(1, 0.00, -0.05), 0.01)
    best_scores = []
    for th in conf_ths:
        try:
            score = scores[confs >= th][-1]
        except IndexError:
            score = 0
        best_scores.append(score)

    return np.array(best_scores), conf_ths


class MetricsEvaluator:
    def __init__(
        self, is_relative_coordinates: bool = True, is_xywh_coordinates: bool = True,
        img_size: Optional[Tuple[int, int]] = None, outpath: str = 'results'
    ) -> None:
        self._type_coordinates = CoordinatesType.Relative if is_relative_coordinates else CoordinatesType.Absolute
        self._box_format = BBFormat.XYWH if is_xywh_coordinates else BBFormat.XYX2Y2
        self._img_size = img_size

        self._all_bounding_boxes = BoundingBoxes()
        self._all_classes = []
        self._outpath = outpath

    def _add_box(self, box: BoundingBox, class_id: str) -> None:
        self._all_bounding_boxes.addBoundingBox(box)
        if class_id not in self._all_classes:
            self._all_classes.append(class_id)

    def add_gt_box(self, img_path: str, class_name: str, bbox: Sequence) -> None:
        bb = BoundingBox(
            img_path,
            class_name,
            *bbox,
            self._type_coordinates,
            self._img_size,
            BBType.GroundTruth,
            format=self._box_format
        )

        self._add_box(bb, class_name)

    def add_det_box(self, img_path: str, class_name: str, bbox: Sequence, confidence: float) -> None:
        bb = BoundingBox(
            img_path,
            class_name,
            *bbox,
            self._type_coordinates,
            self._img_size,
            BBType.Detected,
            confidence,
            format=self._box_format
        )
        self._add_box(bb, class_name)

    def eval_conf_vals(self, iou_th: float = 0.5, beta_f: float = 1) -> list:
        evaluator = Evaluator()

        return evaluator.GetPascalVOCMetrics(
            self._all_bounding_boxes,  # Object containing all bounding boxes (ground truths and detections)
            IOUThreshold=iou_th,  # IOU threshold
            method=MethodAveragePrecision.EveryPointInterpolation,
            betaF=beta_f)

    def eval(self, iou_th: float = 0.5, beta_f: float = 1) -> None:
        evaluator = Evaluator()
        acc_AP = 0
        validClasses = 0
        acc_total_positives = 0
        acc_f1 = 0
        acc_fbeta = 0
        acc_weighted_f1 = 0
        acc_weighted_fbeta = 0
        acc_total_tp = 0
        acc_total_fp = 0

        f_scores_thresholded = []
        precision_thresholded = []
        recall_thresholded = []

        # make_result_folder(self._outpath)

        # Plot Precision x Recall curve
        detections = evaluator.PlotPrecisionRecallCurve(
            self._all_bounding_boxes,  # Object containing all bounding boxes (ground truths and detections)
            IOUThreshold=iou_th,  # IOU threshold
            method=MethodAveragePrecision.EveryPointInterpolation,
            showAP=True,  # Show Average Precision in the title of the plot
            showInterpolatedPrecision=False,  # Don't plot the interpolated precision curve
            savePath=self._outpath,
            showGraphic=False,
            betaF=beta_f)

        f = open(os.path.join(self._outpath, "report.txt"), 'w')
        f.write('Object Detection Metrics\n')
        f.write('https://github.com/rafaelpadilla/Object-Detection-Metrics\n\n\n')
        f.write('Average Precision (AP), Precision and Recall per class:')

        f.write(f'\n\nbeta: {beta_f:.2f}')

        # each detection is a class
        print(f"| {'class':^30} | {'AP':^10} | {'f1':^10} | {'f_beta':^10} |")
        for metricsPerClass in detections:

            # Get metric values per each class
            cl = metricsPerClass['class']
            ap = metricsPerClass['AP']
            precision = metricsPerClass['precision']
            recall = metricsPerClass['recall']
            totalPositives = metricsPerClass['total positives']
            total_TP = metricsPerClass['total TP']
            total_FP = metricsPerClass['total FP']
            confidence = metricsPerClass['confidence']
            f1_score = metricsPerClass['f1_score']
            f_beta_score = metricsPerClass['f_beta_score']

            if totalPositives > 0:
                f1 = f1_score[-1] if len(f1_score) else 0
                fbeta = f_beta_score[-1] if len(f_beta_score) else 0

                validClasses = validClasses + 1
                acc_AP = acc_AP + ap
                acc_f1 += f1
                acc_fbeta += fbeta

                acc_total_positives += totalPositives
                acc_total_tp += total_TP
                acc_total_fp += total_FP
                acc_weighted_f1 += f1 * totalPositives
                acc_weighted_fbeta += fbeta * totalPositives

                best_scores, conf_ths = get_scores_by_conf_th(f_beta_score, confidence)
                best_precision, _ = get_scores_by_conf_th(precision, confidence)
                best_recall, _ = get_scores_by_conf_th(recall, confidence)
                f_scores_thresholded.append(best_scores)
                precision_thresholded.append(best_precision)
                recall_thresholded.append(best_recall)

                ap_str = "{0:.2f}%".format(ap * 100)
                f1_str = "{0:.2f}%".format(f1 * 100)
                fbeta_str = "{0:.2f}%".format(fbeta * 100)
                print(f"| {cl:<30} | {ap_str:>10} | {f1_str:>10} | {fbeta_str:>10} |")
                f.write(f'\n\nClass: {cl}')

                f.write('\nAVERAGE')
                f.write(f'\nAP          : {ap_str}')
                f.write(f'\nF1 score    : {f1_str}')
                f.write(f'\nF_beta score: {fbeta_str}')

                f.write('\nGROUPED')
                f.write(f'\nConfidence ths: {array2str(conf_ths)}')
                f.write(f'\nF_beta score  : {array2str(best_scores)}')
                f.write(f'\nPrecision     : {array2str(best_precision)}')
                f.write(f'\nRecall        : {array2str(best_recall)}')

                f.write('\nALL DETECTIONS')
                f.write(f'\nPrecision     : {array2str(precision)}')
                f.write(f'\nRecall        : {array2str(recall)}')
                f.write(f'\nConfidence    : {array2str(confidence)}')
                f.write(f'\nF1 score      : {array2str(f1_score)}')
                f.write(f'\nF_beta score  : {array2str(f_beta_score)}')

        mAP = acc_AP / (validClasses + 1e-6)

        macro_f1 = acc_f1 / (validClasses + 1e-6)
        macro_fbeta = acc_fbeta / (validClasses + 1e-6)

        weighted_f1 = acc_weighted_f1 / acc_total_positives
        weighted_fbeta = acc_weighted_fbeta / acc_total_positives

        rec = acc_total_tp / acc_total_positives
        prec = np.divide(acc_total_tp, (acc_total_fp + acc_total_tp))
        f1_score = 2 * prec * rec / (prec + rec + 1e-6)
        f_beta_score = (1 + beta_f**2) * prec * rec / (beta_f**2 * prec + rec + 1e-6)

        mAP_str = "{0:.2f}%".format(mAP * 100)
        macro_f1_str = "{0:.2f}%".format(macro_f1 * 100)
        macro_fbeta_str = "{0:.2f}%".format(macro_fbeta * 100)
        weighted_f1_str = "{0:.2f}%".format(weighted_f1 * 100)
        weighted_fbeta_str = "{0:.2f}%".format(weighted_fbeta * 100)
        global_f1_str = "{0:.2f}%".format(f1_score * 100)
        global_fbeta_str = "{0:.2f}%".format(f_beta_score * 100)

        f_scores_thresholded = np.mean(f_scores_thresholded, axis=0)
        recall_thresholded = np.mean(recall_thresholded, axis=0)
        precision_thresholded = np.mean(precision_thresholded, axis=0)
        idxs = np.argsort(f_scores_thresholded)[::-1]

        f_scores_thresholded = f_scores_thresholded[idxs]
        recall_thresholded = recall_thresholded[idxs]
        precision_thresholded = precision_thresholded[idxs]
        conf_ths = conf_ths[idxs]

        print()
        print(f'{"mAP":<20}: {mAP_str:>10}')
        print(f'{"macro avg f1":<20}: {macro_f1_str:>10}')
        print(f'{"macro avg f_beta":<20}: {macro_fbeta_str:>10}')
        print(f'{"weighted avg f1":<20}: {weighted_f1_str:>10}')
        print(f'{"weighted avg f_beta":<20}: {weighted_fbeta_str:>10}')
        print(f'{"global avg f1":<20}: {global_f1_str:>10}')
        print(f'{"global avg f_beta":<20}: {global_fbeta_str:>10}')

        f.write('\n\nSUMMARY')
        f.write(f'\nmAP                : {mAP_str}')
        f.write(f'\nmacro avg f1       : {macro_f1_str}')
        f.write(f'\nmacro avg f_beta   : {macro_fbeta_str}')
        f.write(f'\nweighted avg f1    : {weighted_f1_str}')
        f.write(f'\nweighted avg f_beta: {weighted_fbeta_str}')
        f.write(f'\nglobal avg f1      : {global_f1_str}')
        f.write(f'\nglobal avg f_beta  : {global_fbeta_str}')
        f.write(f'\n\nsorted confidence th          : {array2str(conf_ths)}')
        f.write(f'\nsorted macro avg f_beta scores: {array2str(f_scores_thresholded)}')
        f.write(f'\nsorted precision              : {array2str(precision_thresholded)}')
        f.write(f'\nsorted recall                 : {array2str(recall_thresholded)}')
        f.close()
