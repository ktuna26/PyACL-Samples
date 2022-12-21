# Copyright 2022 Huawei Technologies Co., Ltd
# CREATED:  2022-10-04 13:12:13
# MODIFIED: 2022-12-14 22:48:45
#!/bin/bash

# clone necessary repository
echo "Downloading CRAFT repository"
git clone https://github.com/clovaai/CRAFT-pytorch

# create python virtual environment
python3 -m venv convertPt2Onnx
source convertPt2Onnx/bin/activate

# install necessary python libs
pip3 install --upgrade pip
pip3 install -r requirements.txt

# copy necessary file to repo
cp -r onnx_export.py craft_mlt_25k.pth CRAFT-pytorch

# open repo
cd CRAFT-pytorch

# convert pt model to onnx model
python3 onnx_export.py

# mv onnx model
mv craft.onnx ../

# deactivate venv
deactivate
cd ../
# delete virtual environment
rm -r convertPt2Onnx

# remove unnecessary files
sudo rm -r CRAFT-pytorch/
echo "[MODEL] Conversion Done!"