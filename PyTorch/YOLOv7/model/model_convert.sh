#!/bin/bash

# Clone original yolov7 repo
DIR=./yolov7
if [ -d "$DIR" ];
then
    echo "$DIR directory exists."
else
	echo "[Download] YOLOv7 original repo."
    git clone https://github.com/WongKinYiu/yolov7.git
fi

# Download pretrained model file
FILE=yolov7.pt
if [ -f "$FILE" ]; then
    echo "$FILE exists."
else 
    echo "[Download] YOLOv7 pretrained model."
    wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt --no-check-certificate
fi


cd ./yolov7

# Create virtual python environment and install dependencies
python3 -m venv convert_onnx
source convert_onnx/bin/activate
echo "[Download] pip requirements downloading."
pip3 install --upgrade pip 
pip3 install --upgrade setuptools 
pip3 install -r ../requirements.txt 

# Export PT -> ONNX
echo "[Conversion] PT -> ONNX Conversion started."
python3 export.py --weights ../yolov7.pt --grid --simplify --topk-all 100 --iou-thres 0.5 --conf-thres 0.4 --img-size 640 640 --max-wh 640

# Deactivate virtual environment
deactivate

# Remove unnecessary files
cd .. && rm yolov7.torchscript.ptl && rm yolov7.torchscript.pt
echo "[Conversion] PT -> ONNX Conversion end."
