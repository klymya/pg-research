import cv2
import numpy as np
import tensorflow as tf
print("TensorFlow version: ", tf.__version__)


class TFLiteInferer:
    def __init__(
            self, yolo_pth: str | None = None, head_pth: str | None = None, imgsz: int = 320, conf_th: float = 0.35,
            iou_th = 0.5, idxs: list | None = None
    ) -> None:
        self.yolo_pth = yolo_pth
        if self.yolo_pth is not None:
            self.yolo_interpreter = tf.lite.Interpreter(model_path=self.yolo_pth)
            self.yolo_interpreter.allocate_tensors()

            self.yolo_input_details = self.yolo_interpreter.get_input_details()
            self.yolo_output_details = self.yolo_interpreter.get_output_details()

        # Initialize the interpreter
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

        self.iou_th = iou_th
        self.conf_th = conf_th

    def pre_process(self, img: np.ndarray) -> np.ndarray:
        if img.shape == self.input_shape:
            return img
        im = np.zeros(self.input_shape, dtype=np.uint8)

        original_h, origial_w, _ = img.shape
        ratio = self.img_shape[0] / origial_w
        resized_h = int(original_h * ratio)

        resized_img = cv2.resize(img, (self.img_shape[0], resized_h))
        im[0, :resized_h, :self.img_shape[0], :] = resized_img[:, :, :]

        return im

    def infer_yolo(self, im: np.ndarray) -> list[np.ndarray]:
        if self.yolo_pth is None:
            raise AttributeError("yolo interpreter not initialized")

        details = self.yolo_input_details[0]
        self.yolo_interpreter.set_tensor(details['index'], im)
        self.yolo_interpreter.invoke()
        y = []
        for output in self.yolo_output_details:
            x = self.yolo_interpreter.get_tensor(output['index'])
            scale, zero_point = output['quantization']
            x = (x.astype(np.float32) - zero_point) * scale  # re-scale

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
                    head_output[0][i][0], # - (0.5 * outputs[0][i][2]),
                    head_output[0][i][1], # - (0.5 * outputs[0][i][3]),
                    head_output[0][i][2],
                    head_output[0][i][3],
                ]
                boxes.append(box)
                scores.append(maxScore)
                class_ids.append(maxClassIndex)
                kps.append(head_output[0][i][5:7])

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

    def infer(self, img: np.ndarray):
        return self.post_process(self.infer_head(self.infer_yolo(self.pre_process(img))))
