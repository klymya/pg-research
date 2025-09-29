import os

from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(
    data="coco128.yaml",
    epochs=1,
    imgsz=640,    
    # device="cpu",
    project=os.path.join(os.getcwd(), "results"),
    name="latest_gpu",
    exist_ok=True,
)
print(results)
