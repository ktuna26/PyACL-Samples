# Copyright 2022 Huawei Technologies Co., Ltd
# CREATED:  2022-10-04 13:12:13
# MODIFIED: 2022-12-14 22:48:45
#!/bin/bash

# clone necessary repository
echo "Downloading Deep Text Recognition Benchmark repository"
git clone https://github.com/clovaai/deep-text-recognition-benchmark.git

# create python virtual environment
python3 -m venv convertPt2Onnx
source convertPt2Onnx/bin/activate

# install necessary python libs
pip3 install --upgrade pip
pip3 install -r requirements.txt

# copy necessary file to repo
cp -r onnx_export.py model.py None-ResNet-None-CTC.pth deep-text-recognition-benchmark

# open repo
cd deep-text-recognition-benchmark

# convert pt model to onnx model
python3 onnx_export.py

# mv onnx model
mv None-ResNet-None-CTC.onnx ../

# deactivate venv
deactivate
cd ../
# delete virtual environment
rm -r convertPt2Onnx

# remove unnecessary files
sudo rm -r deep-text-recognition-benchmark/
echo "[MODEL] Conversion Done!"