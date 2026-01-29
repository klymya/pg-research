import os
import argparse

from ultralytics import YOLO


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="baseline")
parser.add_argument("--exp_name", type=str, default="baseline")
parser.add_argument("--cfg", type=str, default="")
parser.add_argument("--weights", type=str, default="yolov8s-pose.pt")
parser.add_argument("--device", nargs="+", type=int, default=[0])
parser.add_argument("--batch", type=int, default=32)
parser.add_argument("--kwargs", type=str, default="")
parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument("--patience", type=int, default=50)
args = parser.parse_args()

if args.cfg == "":
    model = YOLO(args.weights)
else:
    model = YOLO(args.cfg).load(args.weights)

kwargs = eval(args.kwargs) if args.kwargs else {}

# Train the model
results = model.train(
    data=os.path.join(args.dataset, "project.yaml"),
    epochs=args.epochs,
    patience=args.patience,
    imgsz=320,
    batch=args.batch * len(args.device),
    device=args.device,
    project=os.path.join(os.getcwd(), "results"),
    name=args.exp_name,
    exist_ok=True,
    **kwargs,
)
