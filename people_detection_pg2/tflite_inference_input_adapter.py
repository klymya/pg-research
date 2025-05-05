import cv2
import numpy as np
import tensorflow as tf

from tflite_inference import TFLiteInferer


class TFLiteInferenceWithInputAdapter(TFLiteInferer):
    def __init__(
        self, yolo_pth: str | None = None, head_pth: str | None = None, adapter_pth: str | None = None,
        imgsz: int = 320, conf_th: float = 0.35, iou_th = 0.5, idxs: list | None = None
    ) -> None:

        super().__init__(
            yolo_pth=yolo_pth,
            head_pth=head_pth,
            imgsz=imgsz,
            conf_th=conf_th,
            iou_th=iou_th,
            idxs=idxs,
        )
        self.input_adapter = tf.lite.Interpreter(model_path=adapter_pth)
        self.input_adapter.allocate_tensors()

        self.input_adapter_input_details = self.input_adapter.get_input_details()[0]
        self.input_adapter_output_details = self.input_adapter.get_output_details()[0]

    def process(self, img: np.ndarray) -> list[np.ndarray]:
        self.input_adapter.set_tensor(self.input_adapter_input_details['index'], img[None])
        self.input_adapter.invoke()

        y = self.input_adapter.get_tensor(self.input_adapter_output_details['index'])
        scale, zero_point = self.input_adapter_output_details['quantization']
        y = (y.astype(np.float32) - zero_point) * scale  # re-scale
        y = (y * 255).astype(np.uint8)

        return y

    def pre_process(self, img: np.ndarray) -> np.ndarray:
        if img.shape == self.input_shape:
            im = img.copy()

            y = self.process(img[0, 40:-40, :, :])
            im[:, 40:-40, :, :] = y[:, :, :, :]
        else:
            im = np.zeros(self.input_shape, dtype=np.uint8)

            original_h, origial_w, _ = img.shape
            self.ratio = self.img_shape[0] / origial_w
            resized_h = int(original_h * self.ratio)

            resized_img = cv2.resize(img, (self.img_shape[0], resized_h))

            y = self.process(resized_img)

            im[:, :resized_h, :self.img_shape[0], :] = y[:, :, :, :]

        return im
