# Copyright 2022 Huawei Technologies Co., Ltd
# CREATED:  2022-10-04 13:12:13
# MODIFIED: 2022-12-14 22:48:45
#!/bin/bash

# yolov5 model type (s,m,x) 
pt_model=$1

# create python virtual environment
python3 -m venv convertPt2Onnx
source convertPt2Onnx/bin/activate

# copy necessary file to repo
cp ${pt_model} export/

# open repo
cd export/

# install necessary python libs
python -m pip install --upgrade pip
pip3 install -r requirements.txt

# convert pt model to onnx model
python3 onnx_export.py --weights ${pt_model}

# mv onnx model
mv "${pt_model%.*}.onnx"  ../

# deactivate venv
deactivate
cd ../
# delete virtual environment
rm -r convertPt2Onnx

echo "[MODEL] Conversion Done!"