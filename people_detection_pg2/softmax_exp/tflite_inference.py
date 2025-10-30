import os
import numpy as np
import tensorflow as tf


class TFLiteInference:
    def __init__(self, model_path: str, is_uint8: bool = True, imgsz: int = 320, layer_indices: list[int] = None):
        """
        Initialize TFLite model inference.

        Args:
            model_path: Path to the TFLite model
            is_uint8: Whether the model uses uint8 quantization
            imgsz: Input image size (default 320)
            layer_indices: List of layer indices to collect during inference (default: None)
        """
        self.model_path = model_path
        self.is_uint8 = is_uint8
        self.imgsz = imgsz
        self.input_shape = (1, imgsz, imgsz, 3)
        self.layer_indices = layer_indices or []

        # Softmax layer indices
        if is_uint8:
            self.softmax_input_idx = 489
            self.softmax_output_idx = 490
        else:
            self.softmax_input_idx = 483
            self.softmax_output_idx = 484

        # Initialize interpreter
        self.interpreter = tf.lite.Interpreter(
            model_path=model_path,
            experimental_preserve_all_tensors=True,
            experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF,
            experimental_delegates=[],
        )
        self.interpreter.allocate_tensors()

        # Get tensor details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        all_details = self.interpreter.get_tensor_details()
        all_indices = {d["index"] for d in all_details}

        if self.softmax_input_idx not in all_indices or self.softmax_output_idx not in all_indices:
            raise ValueError(
                f"Given indices not found in this model. "
                f"Available index range: [{min(all_indices)}..{max(all_indices)}]. "
                f"Got IN={self.softmax_input_idx}, OUT={self.softmax_output_idx}."
            )

        # Validate layer indices
        for idx in self.layer_indices:
            if idx not in all_indices:
                raise ValueError(
                    f"Layer index {idx} not found in this model. "
                    f"Available index range: [{min(all_indices)}..{max(all_indices)}]."
                )

        # Store logits and probs details
        self.logits_details = next(d for d in all_details if d["index"] == self.softmax_input_idx)
        self.probs_details = next(d for d in all_details if d["index"] == self.softmax_output_idx)

        # Store layer details for requested indices
        self.all_details = all_details
        self.layer_details = {idx: next(d for d in all_details if d["index"] == idx) for idx in self.layer_indices}

        # Storage for last inference results
        self._logits_fp32 = None
        self._probs_fp32 = None
        self._layer_outputs = {}

    def _qparams(self, d):
        """Extract quantization parameters from tensor details."""
        qp = d.get("quantization_parameters", {})
        return qp.get("scales", None), qp.get("zero_points", None)

    def _get_tensor(self, d) -> np.ndarray:
        """Get tensor from interpreter by details."""
        return self.interpreter.get_tensor(d["index"])

    def infer_image(self, img: np.ndarray) -> dict:
        """
        Run inference on an image.

        Args:
            img: Input image as numpy array

        Returns:
            Dictionary containing inference results including logits, probs, and requested layer outputs
        """
        # Preprocess image
        input_data = self._preprocess(img)

        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)

        # Run inference
        self.interpreter.invoke()

        # Get logits and probs
        logits_np = self._get_tensor(self.logits_details)
        probs_np = self._get_tensor(self.probs_details)

        # Dequantize if uint8
        if self.is_uint8:
            logits_scale, logits_zero_point = self._qparams(self.logits_details)
            probs_scale, probs_zero_point = self._qparams(self.probs_details)

            logits_np = (logits_np.astype(np.float32) - logits_zero_point[0]) * logits_scale[0]
            probs_np = (probs_np.astype(np.float32) - probs_zero_point[0]) * probs_scale[0]

        # Store for later retrieval
        self._logits_fp32 = logits_np
        self._probs_fp32 = probs_np

        # Get requested layer outputs
        self._layer_outputs = {}
        for idx in self.layer_indices:
            layer_output = self._get_tensor(self.layer_details[idx])

            # Dequantize if uint8
            if self.is_uint8:
                scale, zero_point = self._qparams(self.layer_details[idx])
                if scale is not None and zero_point is not None:
                    layer_output = (layer_output.astype(np.float32) - zero_point[0]) * scale[0]

            self._layer_outputs[idx] = layer_output

        # Get output tensors
        outputs = []
        for detail in self.output_details:
            output = self.interpreter.get_tensor(detail['index'])

            # Dequantize if uint8
            if self.is_uint8:
                scale, zero_point = self._qparams(detail)
                if scale is not None and zero_point is not None:
                    output = (output.astype(np.float32) - zero_point[0]) * scale[0]

            outputs.append(output)

        return {
            'outputs': outputs,
            'logits': logits_np,
            'probs': probs_np,
            'layers': self._layer_outputs.copy()
        }

    def get_logits_probs(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get logits and probs in fp32 from last inference.

        Returns:
            Tuple of (logits_fp32, probs_fp32)
        """
        if self._logits_fp32 is None or self._probs_fp32 is None:
            raise RuntimeError("No inference has been run yet. Call infer_image() first.")

        return self._logits_fp32, self._probs_fp32

    def get_layer_outputs(self) -> dict[int, np.ndarray]:
        """
        Get layer outputs in fp32 from last inference.

        Returns:
            Dictionary mapping layer indices to their outputs
        """
        if not self._layer_outputs:
            raise RuntimeError("No inference has been run yet. Call infer_image() first.")

        return self._layer_outputs.copy()

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        Preprocess image for model input.

        Args:
            img: Input image

        Returns:
            Preprocessed image array
        """
        import cv2

        if img.shape == self.input_shape:
            return img

        dtype = np.uint8 if self.is_uint8 else np.float32
        im = np.zeros(self.input_shape, dtype=dtype)

        original_h, original_w = img.shape[:2]
        ratio = self.imgsz / original_w
        resized_h = int(original_h * ratio)

        resized_img = cv2.resize(img, (self.imgsz, resized_h))
        im[0, :resized_h, :self.imgsz, :] = resized_img[:, :, :]

        return im
