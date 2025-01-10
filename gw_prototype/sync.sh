rsync -auv \
    --exclude='.__pycache__/' \
    --exclude='.pytest_cache' \
    --exclude='.vscode/' \
    --exclude='*cache*' \
    --exclude='.git/' \
    --exclude='da_viz' \
    --exclude='camera_calibration' \
    --exclude='tests' \
    --exclude='*.pyc' \
    --exclude='*.db' \
    ./ pi0:/home/pi/gw
