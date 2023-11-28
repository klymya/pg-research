from ultralytics import YOLO

model = YOLO("./yolov8-pose.yaml")
model.train(data="positiondb_pose.yaml", epochs=1000, imgsz=640, name="positiondb_pose", batch=-1)  # , cfg="default-norm.yaml")

path = model.export(format="onnx")  # export the model to ONNX format
print(path)
