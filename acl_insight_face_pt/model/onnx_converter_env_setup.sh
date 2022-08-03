mkdir original_repo
cd original_repo/
git clone https://github.com/deepinsight/insightface.git
cd insightface/detection/scrfd

python -m venv onnxenv
source onnxenv/bin/activate

pip install --upgrade pip
pip install numpy
pip install torchvision==0.11.3 torch==1.10.2

pip install -U openmim
mim install mmcv-full==1.2.6

git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .
cd ..
pip install -v -e .
pip install -r requirements/build.txt
pip install -v -e .
pip install onnx onnxruntime onnx-simplifier
