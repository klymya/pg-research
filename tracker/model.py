import numpy as np
import cv2
import tensorflow as tf


class Model:
    def __init__(
        self, model_pth: str, head_pth: str, conf_th: float = 0.25, iou_th: float = 0.5
    ) -> None:
        self.conf_th = conf_th
        self.iou_th = iou_th
        
        self.model_interpreter = tf.lite.Interpreter(model_pth)
        self.model_interpreter.allocate_tensors()

        self.head_interpreter = tf.lite.Interpreter(head_pth)
        self.head_interpreter.allocate_tensors()
        
        self.img_size = self.model_interpreter.get_input_details()[0]["shape"][1]
        self.model_input = np.zeros(self.model_interpreter.get_input_details()[0]["shape"], dtype=np.uint8)
        self.ratio = 1

    def preprocessing(self, img: np.ndarray) -> None:
        h, w, _ = img.shape
        self.ratio = self.img_size / w
        resized_h = int(h * self.ratio)

        resized_img = cv2.resize(img, (self.img_size, resized_h))
        self.model_input[0, :resized_h, :self.img_size, :] = resized_img[:, :, :]

    def inference(self) -> np.ndarray:
        input_details = self.model_interpreter.get_input_details()[0]
        output_details = self.model_interpreter.get_output_details()

        self.model_interpreter.set_tensor(input_details['index'], self.model_input)
        self.model_interpreter.invoke()
        y = []
        for output in output_details:
            x = self.model_interpreter.get_tensor(output['index'])
            scale, zero_point = output['quantization']
            x = (x.astype(np.float32) - zero_point) * scale  # re-scale

            y.append(x)

        input_details = self.head_interpreter.get_input_details()
        output_details = self.head_interpreter.get_output_details()

        for idx, details in enumerate(input_details):
            self.head_interpreter.set_tensor(details['index'], y[idx])

        self.head_interpreter.invoke()

        outputs = self.head_interpreter.get_tensor(output_details[0]['index'])

        return outputs

    def postprocessing(self, outputs: np.ndarray) -> np.ndarray:
        rows = outputs.shape[1]

        boxes = []
        scores = []
        class_ids = []
        kps = []

        # Iterate through output to collect bounding boxes, confidence scores, and class IDs
        for i in range(rows):
            classes_scores = outputs[0][i][4:5]
            (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
            if maxScore >= self.conf_th:
                box = [
                    outputs[0][i][0] / self.ratio, # - (0.5 * outputs[0][i][2]),
                    outputs[0][i][1] / self.ratio, # - (0.5 * outputs[0][i][3]),
                    outputs[0][i][2] / self.ratio,
                    outputs[0][i][3] / self.ratio,
                ]
                boxes.append(box)
                scores.append(maxScore)
                class_ids.append(maxClassIndex)
                kps.append(outputs[0][i][5:7] / self.ratio)

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

    def predict(self, img: np.ndarray) -> np.ndarray:
        self.preprocessing(img)
        outputs = self.inference()
        detections = self.postprocessing(outputs)

        return detections
