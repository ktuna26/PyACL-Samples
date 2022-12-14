# Copyright 2022 Huawei Technologies Co., Ltd
# CREATED:  2022-10-04 13:12:13
# MODIFIED: 2022-12-14 22:48:45
#!/bin/bash

# clone necessary repository
echo "Downloading IndsightFace repository"
git clone https://github.com/deepinsight/insightface.git

# create python virtual environment
python3 -m venv convertPt2Onnx
source convertPt2Onnx/bin/activate

# install necessary python libs
pip3 install --upgrade pip
pip3 install -r requierements.txt

# copy necessary file to repo
cp -r onnx_export.py scrfd_34g.py scrfd_34g.pth insightface/detection/scrfd
cp ../data/test.jpg insightface/detection/scrfd

# open repo
cd insightface/detection/scrfd

# convert pt model to onnx model
python3 onnx_export.py --config scrfd_34g.py --weights scrfd_34g.pth --input_img sample.jpg --simplify

# mv onnx model
mv scrfd_34g_shape640x640.onnx ../../../

# deactivate venv
deactivate
cd ../../../
# delete virtual environment
rm -r convertPt2Onnx

# remove unnecessary files
rm -r insightface/
echo "[MODEL] Conversion Done!"