import os
from ultralytics import YOLO

DATASET = "no_occupied_office_chairs"
EXP_NAME = "no_occupied_office_chairs"

model = YOLO("yolov8n-pose.yaml").load(
    "/home/klymy/projects/researches/theodore/yolo_artifacts/baseline_ultralytics/weights/best.pt"
)
# model = YOLO("yolov8n-pose.yaml").load("yolov8n.pt")
# model = YOLO("yolov8n-pose.yaml")
# model = YOLO("yolov8n-pose.yaml").load("yolov8n-pose.pt")
# Train the model
results = model.train(
    data=os.path.join("experiments", DATASET, "project.yaml"),
    epochs=1000,
    imgsz=640,
    # batch=33 * 3,
    batch=30 * 3,
    device=[0, 1, 2],
    project=os.path.join(os.getcwd(), "results"),
    name=EXP_NAME,
    exist_ok=True,
)
