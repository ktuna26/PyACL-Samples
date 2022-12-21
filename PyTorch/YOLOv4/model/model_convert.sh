# Copyright 2022 Huawei Technologies Co., Ltd
# CREATED:  2022-10-04 13:12:13
# MODIFIED: 2022-12-14 22:48:45
#!/bin/bash

# create python virtual environment
python3 -m venv convertPt2Onnx
source convertPt2Onnx/bin/activate

# install necessary python libs
python -m pip install --upgrade pip
pip3 install -r requirements.txt

# copy necessary file to repo
cp -r yolov4.pth export/
cp ../data/person.jpg export/

# open repo
cd export/

# convert pt model to onnx model
python3 onnx_export.py yolov4.pth person.jpg 80 608 608

# mv onnx model
mv yolov4.onnx ../

# remove unnecessary files
rm -r yolov4.pth person.jpg

# deactivate venv
deactivate
cd ../
# delete virtual environment
rm -r convertPt2Onnx

echo "[MODEL] Conversion Done!"