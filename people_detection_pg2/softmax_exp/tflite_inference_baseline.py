import json

import cv2
import numpy as np
import tensorflow as tf
print("TensorFlow version: ", tf.__version__)


class TFLiteInferer:
    def __init__(
            self, yolo_pth: str | None = None, head_pth: str | None = None, imgsz: int = 320, conf_th: float = 0.35,
            iou_th = 0.5, idxs: list | None = None, separated_head: bool = False, head_int8: bool = False, bias_pth: str = None
    ) -> None:
        self.yolo_pth = yolo_pth
        if self.yolo_pth is not None:
            self.yolo_interpreter = tf.lite.Interpreter(model_path=self.yolo_pth)
            self.yolo_interpreter.allocate_tensors()

            self.yolo_input_details = self.yolo_interpreter.get_input_details()
            self.yolo_output_details = self.yolo_interpreter.get_output_details()

        # Initialize the interpreter
        self.separated_head = separated_head
        if self.separated_head:
            self.head_processing_func = self.infer_head_sep
        else:
            self.head_processing_func = self.infer_head

        self.head_int8 = head_int8
        self.head_pth = head_pth
        if self.head_pth is not None:
            self.head_interpreter = tf.lite.Interpreter(model_path=head_pth)
            self.head_interpreter.allocate_tensors()

            self.head_input_details = self.head_interpreter.get_input_details()
            self.head_output_details = self.head_interpreter.get_output_details()

            if idxs:
                self.head_idxs = idxs
            else:
                self.head_idxs = list(range(len(self.head_input_details)))

        self.img_shape = (imgsz, imgsz)
        self.input_shape = (1, imgsz, imgsz, 3)
        self.ratio = 1

        self.iou_th = iou_th
        self.conf_th = conf_th

        if bias_pth is not None:
            with open(bias_pth, 'r') as f:
                biases = json.load(f)
            # coefficients and intercepts for bias correction
            self.dx = (biases["dx"]["coef"], biases["dx"]["intercept"])
            self.dy = (biases["dy"]["coef"], biases["dy"]["intercept"])
            self.dw = (biases["dw"]["coef"], biases["dw"]["intercept"])
            self.dh = (biases["dh"]["coef"], biases["dh"]["intercept"])
            self.dx_coord = (biases["dx_coord"]["coef"], biases["dx_coord"]["intercept"])
            self.dy_coord = (biases["dy_coord"]["coef"], biases["dy_coord"]["intercept"])
        else:
            self.dx = ([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.0)
            self.dy = ([0.0, 1.0, 0.0, 0.0, 0.0, 0.0], 0.0)
            self.dw = ([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], 0.0)
            self.dh = ([0.0, 0.0, 0.0, 1.0, 0.0, 0.0], 0.0)
            self.dx_coord = ([0.0, 0.0, 0.0, 0.0, 1.0, 0.0], 0.0)
            self.dy_coord = ([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], 0.0)

    def pre_process(self, img: np.ndarray) -> np.ndarray:
        if img.shape == self.input_shape:
            return img
        im = np.zeros(self.input_shape, dtype=np.uint8)

        original_h, origial_w, _ = img.shape
        self.ratio = self.img_shape[0] / origial_w
        resized_h = int(original_h * self.ratio)

        resized_img = cv2.resize(img, (self.img_shape[0], resized_h))
        im[0, :resized_h, :self.img_shape[0], :] = resized_img[:, :, :]

        return im

    def infer_yolo(self, im: np.ndarray, quantized_output: bool = False) -> list[np.ndarray]:
        if self.yolo_pth is None:
            raise AttributeError("yolo interpreter not initialized")

        details = self.yolo_input_details[0]
        self.yolo_interpreter.set_tensor(details['index'], im)
        self.yolo_interpreter.invoke()
        y = []
        for output in self.yolo_output_details:
            x = self.yolo_interpreter.get_tensor(output['index'])
            if not quantized_output:
                scale, zero_point = output['quantization']
                x = (x.astype(np.float32) - zero_point) * scale  # re-scale
            else:
                x = x.astype(np.int32)

            y.append(x)

        y = [x if isinstance(x, np.ndarray) else x.numpy() for x in y]
        return y

    def infer_head(self, y: list[np.ndarray]) -> np.ndarray:
        if self.head_pth is None:
            raise AttributeError("head interpreter not initialized")

        for idx, details in enumerate(self.head_input_details):
            self.head_interpreter.set_tensor(details['index'], y[self.head_idxs[idx]])

        self.head_interpreter.invoke()

        outputs = self.head_interpreter.get_tensor(self.head_output_details[0]['index'])
        return outputs

    def infer_head_sep(self, y: list[np.ndarray]) -> np.ndarray:
        if self.head_pth is None:
            raise AttributeError("head interpreter not initialized")

        for details in self.head_input_details:
            idx = TFLiteInferer.idx_from_name(details['name'])
            model_input = y[self.head_idxs[idx]].astype(np.float32)
            if self.head_int8:
                scale, zero_point = details['quantization']
                model_input = (model_input / scale + zero_point).astype(np.uint8)

            self.head_interpreter.set_tensor(details['index'], model_input)

        self.head_interpreter.invoke()

        head_outputs = []
        for output in self.head_output_details:
            x = self.head_interpreter.get_tensor(output['index'])
            if self.head_int8:
                scale, zero_point = output['quantization']
                x = (x.astype(np.float32) - zero_point) * scale  # re-scale
            else:
                x = x.astype(np.float32)

            head_outputs.append(x)

        if self.head_int8:
            scores = head_outputs[0]
            bboxes = head_outputs[1]
        else:
            scores = head_outputs[1]
            bboxes = head_outputs[0]

        outputs = np.concatenate([bboxes[:, :, :4], scores, bboxes[:, :, 4:]], axis=-1)
        return outputs

    def post_process(self, head_output: np.ndarray) -> np.ndarray:
        rows = head_output.shape[1]

        boxes = []
        scores = []
        class_ids = []
        kps = []

        # Iterate through output to collect bounding boxes, confidence scores, and class IDs
        for i in range(rows):
            classes_scores = head_output[0][i][4:5]
            (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
            if maxScore >= self.conf_th:
                box = [
                    head_output[0][i][0] / self.ratio, # - (0.5 * outputs[0][i][2]),
                    head_output[0][i][1] / self.ratio, # - (0.5 * outputs[0][i][3]),
                    head_output[0][i][2] / self.ratio,
                    head_output[0][i][3] / self.ratio,
                ]
                kp = [
                    head_output[0][i][5] / self.ratio,
                    head_output[0][i][6] / self.ratio,
                ]
                scores.append(maxScore)
                class_ids.append(maxClassIndex)

                box_corrected = [
                    sum([a * b for a, b in zip(box, self.dx[0][:4])] + [a * b for a, b in zip(kp, self.dx[0][4:])]) + self.dx[1],
                    sum([a * b for a, b in zip(box, self.dy[0][:4])] + [a * b for a, b in zip(kp, self.dy[0][4:])]) + self.dy[1],
                    sum([a * b for a, b in zip(box, self.dw[0][:4])] + [a * b for a, b in zip(kp, self.dw[0][4:])]) + self.dw[1],
                    sum([a * b for a, b in zip(box, self.dh[0][:4])] + [a * b for a, b in zip(kp, self.dh[0][4:])]) + self.dh[1],
                ]
                kp_corrected = [
                    sum([a * b for a, b in zip(box, self.dx_coord[0][:4])] + [a * b for a, b in zip(kp, self.dx_coord[0][4:])]) + self.dx_coord[1],
                    sum([a * b for a, b in zip(box, self.dy_coord[0][:4])] + [a * b for a, b in zip(kp, self.dy_coord[0][4:])]) + self.dy_coord[1],
                ]
                boxes.append(box_corrected)
                kps.append(kp_corrected)

        # Apply NMS (Non-maximum suppression)
        result_boxes = cv2.dnn.NMSBoxes(boxes, scores, self.conf_th, self.iou_th, 0.5)

        detections = []

        # Iterate through NMS results to draw bounding boxes and labels
        for i in range(len(result_boxes)):
            index = result_boxes[i]
            box = boxes[index]

            detections.append([*box, scores[index], class_ids[index], *kps[index]])

        detections = np.array(detections)
        return detections

    def infer(self, img: np.ndarray, quantized_output: bool = False):
        return self.post_process(
            self.head_processing_func(self.infer_yolo(self.pre_process(img), quantized_output=quantized_output)))

    def infer_no_head(self, img: np.ndarray, quantized_output: bool = False) -> list[np.ndarray]:
        return self.infer_yolo(self.pre_process(img), quantized_output=quantized_output)

    def infer_bottleneck(self, img: np.ndarray) -> list[np.ndarray]:
        scores, bboxes = self.infer_yolo(self.pre_process(img))
        return self.post_process(self.merge_scores_bboxes(scores, bboxes))

    @staticmethod
    def idx_from_name(name: str) -> int:
        """
        Extract index from input name.
        """
        return int(name.split('_')[-1].replace(':0', ''))

    @staticmethod
    def merge_scores_bboxes(scores: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
        """
        Merge scores and bounding boxes into a single array.
        """
        return np.concatenate([bboxes[:, :, :4], scores, bboxes[:, :, 4:]], axis=-1)


class TFLiteMerged(TFLiteInferer):
    def __init__(
            self, yolo_pth: str | None = None, imgsz: int = 320, conf_th: float = 0.35,
            iou_th = 0.5, idxs: list | None = None, separated_head: bool = False, head_int8: bool = False, bias_pth: str = None
    ) -> None:
        super().__init__(
            yolo_pth=yolo_pth, head_pth=None, imgsz=imgsz, conf_th=conf_th, iou_th=iou_th, idxs=idxs,
            separated_head=separated_head, head_int8=head_int8, bias_pth=bias_pth,
        )

    def infer(self, img: np.ndarray):
        return self.infer_bottleneck(img)
