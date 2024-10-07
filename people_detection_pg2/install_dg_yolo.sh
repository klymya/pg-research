git clone https://github.com/DeGirum/ultralytics_yolov8.git
cd ultralytics_yolov8 
git checkout 75cab2e0c68723d4344c69a3bcd85265a582ab3d
pip install -e .

cp ../exporter.py ultralytics/engine/exporter.py