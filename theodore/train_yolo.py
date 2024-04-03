import os

from ultralytics import YOLO


FILE_DIR = os.path.dirname(os.path.abspath(__file__))

model = YOLO("yolov8n.pt")
model.train(
    data=os.path.join(FILE_DIR, "theodore_yolo/project.yaml"),
    epochs=1000,
    project=os.path.join(FILE_DIR, "yolo_artifacts"),
    name="baseline",
    device=0,
    cache=True,
    batch=-1,
    exist_ok=True
)
