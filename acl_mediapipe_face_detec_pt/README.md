# PyTorch Mediapipe Face Detection (BlazeFace)
Please open the `jupyter-notebook` for a quick demo!

<<<<<<< HEAD
BlazeFace is a fast, light-weight face detector from Google Research | [Read More](https://sites.google.com/view/perception-cv4arvr/blazeface) | [Paper](https://arxiv.org/abs/1907.05047) | [Pretrained Model](https://github.com/google/mediapipe/blob/v0.7.12/mediapipe/models/face_detection_back.tflite)
=======
BlazeFace is a fast, light-weight face detector from Google Research | [Read More](https://sites.google.com/view/perception-cv4arvr/blazeface) | [Paper](https://arxiv.org/abs/1907.05047) | [Pretrained Model](https://github.com/google/mediapipe/raw/v0.7.12/mediapipe/models/face_detection_back.tflite)
>>>>>>> dev2

<img alt="teaser" src="./figures/mediapipe_small.png">

## Overview
`PyTorch` implementation for **Mediapipe** face detector that effectively detect face area by exploring 6 keypoints (2x eyes, 2x ears, nose, mouth) for face landmarks.
The BlazePaper paper mentions that there are two versions of the model, one for the front-facing camera and one for the back-facing camera. This repo includes only the backend camera model

<img alt="teaser" src="./figures/face_detection_android_gpu.gif">

## Getting started
<<<<<<< HEAD
=======

| **Model** | **CANN Version** | **How to Obtain** |
|---|---|---|
| BlazeFace| 5.1.RC2  | Download pretrained model [BlazeFaceBack](https://github.com/google/mediapipe/raw/v0.7.12/mediapipe/models/face_detection_back.tflite)

Download the [infer-modelzoo(5.1.RC2)](https://ascendhub.huawei.com/#/detail/infer-modelzoo) docker image.

Start your docker environment.

```bash
sudo docker run -it -u root --name mediapipeInfer -p 6565:4545 \
--device=/dev/davinci0 \
--device=/dev/davinci_manager \
--device=/dev/devmm_svm \
--device=/dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /PATH/pyacl_samples:/workspace/pyacl_samples \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
ascendhub.huawei.com/public-ascendhub/infer-modelzoo:22.0.RC2 /bin/bash
```
<details> <summary> Change Docker Python version (<i>click to expand</i>)</summary>

```bash
rm -rf /usr/local/python3.9.2

wget https://www.python.org/ftp/python/3.7.5/Python-3.7.5.tgz --no-check-certificate && \
    tar -zxvf Python-3.7.5.tgz && \
    cd Python-3.7.5 && \
    ./configure --prefix=/usr/local/python3.7.5 --enable-loadable-sqlite-extensions --enable-shared && make && make install && \
    cd .. && \
    rm -r -d Python-3.7.5 && rm Python-3.7.5.tgz && \
    export LD_LIBRARY_PATH=/usr/local/python3.7.5/lib:$LD_LIBRARY_PATH && \
    export PATH=/usr/local/python3.7.5/bin:$PATH
```
```bash
apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        make \
        cmake \
        zlib1g \
        zlib1g-dev \
        openssl \
        libsqlite3-dev \
        libssl-dev \
        libffi-dev \
        unzip \
        pciutils \
        net-tools \
        libblas-dev \
        gfortran \
        libblas3 \
        libopenblas-dev \
        libbz2-dev \
        build-essential \
        git \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
```
```bash
wget https://www.python.org/ftp/python/3.7.5/Python-3.7.5.tgz --no-check-certificate && \
    tar -zxvf Python-3.7.5.tgz && \
    cd Python-3.7.5 && \
    ./configure --prefix=/usr/local/python3.7.5 --enable-loadable-sqlite-extensions --enable-shared && make && make install && \
    cd .. && \
    rm -r -d Python-3.7.5 && rm Python-3.7.5.tgz && \
    export LD_LIBRARY_PATH=/usr/local/python3.7.5/lib:$LD_LIBRARY_PATH && \
    export PATH=/usr/local/python3.7.5/bin:$PATH

pip3 install --upgrade pip

pip3 install attrs numpy decorator sympy cffi pyyaml pathlib2 psutil protobuf scipy requests absl-py jupyter jupyterlab sympy
```
</details>

Convert your model.

### TFLite -> PT model -> ONNX format -> MindSpore .ckpt -> Mindspore .air -> Ascend .om format

For this stages you should use the docker environment. The model_convert.sh file will do every model conversion stage automatically. After conversion you should have the .air model in your /model path.

```bash
cd /workspace/pyacl_samples/acl_mediapipe_face_detec_pt/model

bash model_convert.sh
```

```bash
atc --model=air_blazeface_back.air --framework=1 --output=51RC2blazefaceback_air --soc_version=Ascend310
```
>>>>>>> dev2
Install dependencies;
- opencv-python-headless
- numpy
- Pillow

```
<<<<<<< HEAD
pip install -r requirements.txt
```
And then download the `TFLite` file of from the link.

### TFLite -> PT model -> ONNX format -> Ascend om format
#### TFLite -> PT 
BlazeFace is designed for use on mobile devices, that's why the pretrained model is in TFLite format. 
Follow the [BalzeFace in Python](https://github.com/hollance/BlazeFace-PyTorch) github repository to convert from `TFLite` format to `Pytorch` format.

#### PT -> ONNX
Use in the [BalzeFace in Python](https://github.com/hollance/BlazeFace-PyTorch) repository  the `model/onnx_export.py` the script to convert `PT` file to `ONNX ` file and don't forget copy necessary code snippet from this file into to blazeface.py file.

#### ONNX -> OM
And then use in the same directory atc tool to convert `ONNX ` file to `OM` file as as follows.
```bash
atc --model=blazefaceback.onnx \
    --framework=5 \
    --output=blazefaceback \
    --soc_version=Ascend310 \
    --precision_mode=allow_fp32_to_fp16
```

Finaly, open `jupyter-notebook` and run the code for demo
=======
pip3 install -r requirements.txt
```

Finaly, open `jupyter-notebook` and run the code for demo

```bash
jupyter-lab --port 4545 --ip 0.0.0.0 --no-browser --allow-root
```

Jupyter-lab will open in (localhost):6565
>>>>>>> dev2

## Citation
```
@inproceedings{GoogleResarch2019BlazeFace,
  title={BlazeFace: Sub-millisecond Neural Face Detection on Mobile GPUs},
  author={Valentin Bazarevsky, Yury Kartynnik, Andrey Vakunov, Karthik Raveendran, Matthias Grundmann},
  booktitle={arXiv:1907.05047v1 [cs.CV] 11 Jul 2019},
  pages={9365--9374},
  year={2019}
}
```