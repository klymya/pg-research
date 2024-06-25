import os

import ultralytics


MODEL_NAME = "v2_batch6_merged_furniture"


model = ultralytics.YOLO(os.path.join("results", MODEL_NAME, "weights", "best.pt"))
model.export(format="onnx")
