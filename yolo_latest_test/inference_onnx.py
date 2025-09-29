import cv2
import numpy as np
import onnxruntime as ort
import os
from typing import List, Tuple

# COCO class names
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

def preprocess_image(image: np.ndarray, input_size: Tuple[int, int] = (640, 640)) -> Tuple[np.ndarray, float, Tuple[int, int], Tuple[int, int]]:
    """Preprocess image for YOLO inference"""
    original_shape = image.shape[:2]

    # Resize while maintaining aspect ratio
    h, w = original_shape
    scale = min(input_size[0] / h, input_size[1] / w)
    new_h, new_w = int(h * scale), int(w * scale)

    resized = cv2.resize(image, (new_w, new_h))

    # Pad to target size
    padded = np.full((input_size[0], input_size[1], 3), 114, dtype=np.uint8)
    padded[:new_h, :new_w] = resized

    # Convert to RGB and normalize
    padded = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    padded = padded.astype(np.float32) / 255.0

    # Transpose to CHW format
    padded = np.transpose(padded, (2, 0, 1))
    padded = np.expand_dims(padded, axis=0)

    return padded, scale, (new_h, new_w), original_shape

def postprocess_detections(outputs: np.ndarray, scale: float, conf_threshold: float = 0.5, nms_threshold: float = 0.4) -> List[dict]:
    """Post-process YOLO outputs to get final detections"""
    detections = []

    # YOLO11 output format: [batch, 84, 8400] where 84 = 4 (bbox) + 80 (classes)
    output = outputs[0]  # Remove batch dimension
    output = output.transpose(1, 0)  # [8400, 84]

    boxes = output[:, :4]  # x_center, y_center, width, height
    scores = output[:, 4:].max(axis=1)  # Max class score
    class_ids = output[:, 4:].argmax(axis=1)  # Class with max score

    # Filter by confidence
    valid_indices = scores > conf_threshold
    boxes = boxes[valid_indices]
    scores = scores[valid_indices]
    class_ids = class_ids[valid_indices]


    if len(boxes) == 0:
        return detections

    # Convert normalized coordinates to pixel coordinates
    # YOLO outputs are normalized (0-1), so multiply by input size (640)
    x_center, y_center, width, height = boxes[:, 0] * 640, boxes[:, 1] * 640, boxes[:, 2] * 640, boxes[:, 3] * 640

    # Convert center format to corner format
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2

    # Scale back to original image size
    x1 = x1 / scale
    y1 = y1 / scale
    x2 = x2 / scale
    y2 = y2 / scale

    # Prepare boxes for NMS (in x1, y1, w, h format)
    nms_boxes = []
    for i in range(len(x1)):
        nms_boxes.append([x1[i], y1[i], x2[i] - x1[i], y2[i] - y1[i]])

    # Apply NMS
    indices = cv2.dnn.NMSBoxes(
        nms_boxes, scores.tolist(), conf_threshold, nms_threshold
    )

    if len(indices) > 0:
        indices = indices.flatten()
        for i in indices:
            detections.append({
                'bbox': [int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])],
                'confidence': float(scores[i]),
                'class_id': int(class_ids[i]),
                'class_name': COCO_CLASSES[class_ids[i]]
            })

    return detections

def draw_detections(image: np.ndarray, detections: List[dict]) -> np.ndarray:
    """Draw bounding boxes and labels on image"""
    result_image = image.copy()

    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        confidence = det['confidence']
        class_name = det['class_name']

        # Draw bounding box
        cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw label
        label = f"{class_name}: {confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(result_image, (x1, y1 - label_size[1] - 10),
                     (x1 + label_size[0], y1), (0, 255, 0), -1)
        cv2.putText(result_image, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    return result_image

def main():
    # Paths
    model_path = "yolov8n.onnx"
    image_path = "datasets/coco128/images/train2017/000000000009.jpg"
    output_path = "inference_result.jpg"

    # Load ONNX model
    print("Loading ONNX model...")
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    print(f"Model input shape: {input_shape}")

    # Load and preprocess image
    print(f"Loading image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    original_image = image.copy()
    preprocessed, scale, new_shape, original_shape = preprocess_image(image)

    # Run inference
    print("Running inference...")
    outputs = session.run(None, {input_name: preprocessed})

    # Post-process results
    print("Post-processing results...")
    detections = postprocess_detections(outputs[0], scale)

    # Draw results
    result_image = draw_detections(original_image, detections)

    # Save result
    cv2.imwrite(output_path, result_image)
    print(f"Results saved to: {output_path}")
    print(f"Found {len(detections)} detections:")

    for i, det in enumerate(detections):
        print(f"  {i+1}: {det['class_name']} ({det['confidence']:.3f}) at {det['bbox']}")

if __name__ == "__main__":
    main()