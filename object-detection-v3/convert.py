import os

import ultralytics


MODEL_NAME = "v3_2_table_chair_table_top_v2_no_occupied_chair"


model = ultralytics.YOLO(os.path.join("results", MODEL_NAME, "weights", "best.pt"))
model.export(format="onnx")
