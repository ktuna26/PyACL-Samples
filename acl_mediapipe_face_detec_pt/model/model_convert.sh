#!/bin/bash
echo "Downloading BlazeFace repository"

git clone https://github.com/hollance/BlazeFace-PyTorch.git

# Create python virtual environment
python3 -m venv convertTfliteOnnx

source convertTfliteOnnx/bin/activate

pip3 install --upgrade pip

pip3 install -r requierements.txt

cp tflite_pthExport.py BlazeFace-PyTorch/
cp onnx_export.py BlazeFace-PyTorch/

cd BlazeFace-PyTorch/

wget https://github.com/google/mediapipe/raw/v0.7.12/mediapipe/models/face_detection_back.tflite --no-check-certificate

python3 tflite_pthExport.py

python3 onnx_export.py

mv blazefaceback.onnx ../

deactivate


cd ..
# Delete virtual environment
rm -r convertTfliteOnnx


# Install Mindspore-ascend-310-x86
pip3 install https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.8.1/MindSpore/ascend/x86_64/mindspore_ascend-1.8.1-cp37-cp37m-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple

# MindConverter

pip3 install onnx~=1.8.0
pip3 install onnxoptimizer~=0.1.2
pip3 install onnxruntime~=1.5.2
pip3 install protobuf==3.20.0

pip3 install torch==1.8.2+cpu -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
pip3 install https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.7.0/MindInsight/any/mindconverter-1.7.0-py3-none-any.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple

mindconverter --model_file blazefaceback.onnx

cp ms_export.py output/

cd output

python3 ms_export.py

mv air_blazeface_back.air ../

echo "[MODEL] Conversion Done!"
