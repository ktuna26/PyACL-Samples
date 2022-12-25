# Copyright 2022 Huawei Technologies Co., Ltd
# CREATED:  2022-10-04 13:12:13
# MODIFIED: 2022-12-23 22:48:45
#!/bin/bash

# prepering environment
update-ca-certificates --fresh
export SSL_CERT_DIR=/etc/ssl/certs

# create python virtual environment
echo "[ENV] Virtual Environment Preparation Starting!"
python3 -m venv convertTfOnnx
source convertTfOnnx/bin/activate

# install necessary python libs
python -m pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet 
echo "[ENV] Virtual Environment Preparation Done!"

# convert pt model to onnx model
echo "[MODEL] Conversion Starting!"
model_name='arcface'
python onnx_export.py --model $model_name --output $model_name

# deactivate venv
deactivate

# Delete virtual environment
rm -r convertTfOnnx

echo "[MODEL] Conversion Done!"