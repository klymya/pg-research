import abc

import cv2
import numpy as np

CLASSES = {
    0: 'person',
    1: 'chair',
    2: 'table',
    3: 'armchair',
    4: 'tv',
    5: 'wheeled walker'
}

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


class BaseModel(abc.ABC):
    def __init__(self,  onnx_model, img_size, **kwargs):
        self.model = cv2.dnn.readNetFromONNX(onnx_model)
        self.img_size = img_size

    @abc.abstractmethod
    def preprocessing(self, img):
        pass

    @abc.abstractmethod
    def postprocessing(self, outputs):
        pass

    def infer(self, img):
        blob = self.preprocessing(img)

        # Perform inference
        self.model.setInput(blob)
        outputs = self.model.forward()

        return self.postprocessing(outputs)


class Detector(BaseModel):
    def __init__(self,  onnx_model, img_size=640, **kwargs):
        super().__init__(onnx_model, img_size, **kwargs)

        self.classes = kwargs.get("classes", CLASSES)
        self.conf_th = kwargs.get("conf_th", 0.2)
        self.iou_th = kwargs.get("iou_th", 0.45)
        self.scale = 1

    def preprocessing(self, img):
        [height, width, _] = img.shape

        # Prepare a square image for inference
        length = max((height, width))
        image = np.zeros((length, length, 3), np.uint8)
        image[0:height, 0:width] = img

        # Calculate scale factor
        self.scale = length / self.img_size

        # Preprocess the image and prepare blob for model
        blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(self.img_size, self.img_size), swapRB=False)

        return blob

    def postprocessing(self, outputs):
        # Prepare output array
        outputs = np.array([cv2.transpose(outputs[0])])
        rows = outputs.shape[1]

        boxes = []
        scores = []
        class_ids = []

        # Iterate through output to collect bounding boxes, confidence scores, and class IDs
        for i in range(rows):
            classes_scores = outputs[0][i][4:]
            (_, maxScore, _, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
            if maxScore >= self.conf_th:
                box = [
                    outputs[0][i][0] - (0.5 * outputs[0][i][2]),
                    outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                    outputs[0][i][2],
                    outputs[0][i][3],
                ]
                boxes.append(box)
                scores.append(maxScore)
                class_ids.append(maxClassIndex)

        # Apply NMS (Non-maximum suppression)
        result_boxes = cv2.dnn.NMSBoxes(boxes, scores, self.conf_th, self.iou_th, 0.5)

        detections = []

        # Iterate through NMS results to draw bounding boxes and labels
        for index in result_boxes:
            box = boxes[index]
            detection = {
                "class_id": class_ids[index],
                "class_name": CLASSES[class_ids[index]],
                "confidence": scores[index],
                "box": [int(i * self.scale) for i in box]
            }
            detections.append(detection)

        return detections


class Segmentator(BaseModel):
    def __init__(self, onnx_model, img_size=128, **kwargs):
        super().__init__(onnx_model, img_size, **kwargs)
        self.original_w = self.img_size
        self.original_h = self.img_size

    def preprocessing(self, img):
        self.original_h, self.original_w, _ = img.shape

        x = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_CUBIC)
        x = x / 255.0
        x = (x - MEAN) / STD
        return np.transpose(x, (2, 0, 1))[None].astype(np.float32)

    def postprocessing(self, outputs):
        return cv2.resize(outputs[0][0], (self.original_w, self.original_h), interpolation=cv2.INTER_LINEAR_EXACT)
