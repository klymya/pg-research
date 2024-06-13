import os
import argparse

from ultralytics import YOLO


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="baseline")
parser.add_argument("--exp_name", type=str, default="baseline")
parser.add_argument("--weights", type=str, default="yolov8n-pose.pt")
parser.add_argument("--batch", type=int, default=32)
args = parser.parse_args()

model = YOLO("yolov8n-pose.yaml").load(args.weights)

# Train the model
results = model.train(
    data=os.path.join("datasets", args.dataset, "project.yaml"),
    epochs=1000,
    imgsz=640,
    batch=args.batch * 3,
    device=[0, 1, 2],
    project=os.path.join(os.getcwd(), "results"),
    name=args.exp_name,
    exist_ok=True,
)
