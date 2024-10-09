git clone https://github.com/DeGirum/ultralytics_yolov8.git
cd ultralytics_yolov8 
git checkout 75cab2e0c68723d4344c69a3bcd85265a582ab3d
pip install -e .

cp ../patch/exporter.py ultralytics/engine/exporter.py
cp ../patch/predictor.py ultralytics/engine/predictor.py 
cp ../patch/predict.py ultralytics/models/yolo/pose/predict.py 
cp ../patch/autobackend.py ultralytics/nn/autobackend.py
cp ../patch/plotting.py ultralytics/utils