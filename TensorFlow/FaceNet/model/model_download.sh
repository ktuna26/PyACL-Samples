# Copyright 2022 Huawei Technologies Co., Ltd
# CREATED:  2023-01-04 03:45:13
# MODIFIED: 2023-01-04 03:50:32
#!/bin/bash

echo "[MODEL] Download Started!"
# Download PB pretrained model
wget https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/modelzoo/Official/cv/Facenet_for_ACL.zip --no-check-certificate

# Unzip the model file
unzip Facenet_for_ACL.zip

# Copy PB model to ./model directory
cp Facenet_for_ACL/facenet_tf.pb .

# Remove unnecessary files and folders.
rm -r Facenet_for_ACL Facenet_for_ACL.zip
echo "[MODEL] Download Complated!"