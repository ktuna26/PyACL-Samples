# InsightFace : SCRFD Face Detection

Please open the `jupyter-notebook` for a quick demo | [Pretrained Model](https://onebox.huawei.com/p/d2bb0e04156ba1c43091f7d8946eb293) | [Paper](https://arxiv.org/abs/2105.04714) | [Original Github Repository](https://github.com/deepinsight/insightface/tree/master/detection/scrfd)

<div align="left">
  <img src="./figures/insight_face.jpg" width="240" alt="prcurve"/>
</div>

## Overview
`SCRFD` is an efficient high accuracy face detection approach which 

<img src="./figures/scrfd_evelope.jpg" width="400" alt="prcurve"/>

## Getting started

Download following **SCRFD PT model** from the link and put it in the model folder by changing its name as **scrfd_34g**. 

| **Model** | **CANN Version** | **How to Obtain** |
|---|---|---|
| SCRFD | 5.1.RC2  | Download pretrained model [SCRFD_34G](https://onedrive.live.com/?authkey=%21AJnAV5tWWaU6%5FPM&id=4A83B6B633B029CC%215538&cid=4A83B6B633B029CC)

<details> <summary> Working on docker environment (<i>click to expand</i>)</summary>

Start your docker environment.

```bash
sudo docker run -it -u root --rm --name mediapipeInfer -p 6565:4545 \
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

```bash
pip3 install --upgrade pip
pip3 install attrs numpy decorator sympy cffi pyyaml pathlib2 psutil protobuf scipy requests absl-py jupyter jupyterlab sympy
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
</details>

## Convert Your Model

### PT model -> ONNX format -> Ascend om format

For this stages it is recommended to use the docker environment to avoid affecting the development environment. The model_convert.sh file will do model conversion stage automatically. After conversion you should have the .onnx model in your /model path.

```bash
cd <root_path_of_pyacl_samples>/pyacl_samples/PyTorch/acl_insight_face/model

bash model_convert.sh
```

```bash
atc --model=scrfd_34g_shape640x640.onnx \
    --framework=5 \
    --output=scrfd_34g \
    --soc_version=Ascend310 \
    --precision_mode=allow_fp32_to_fp16
```

Install dependencies;
- opencv-python>=3.4.2
-  Pillow
- numpy

```
pip3 install -r requirements.txt
```

Finaly, open `jupyter-notebook` and run the code for demo

```bash
jupyter-notebook --port 4545 --ip 0.0.0.0 --no-browser --allow-root
```

Jupyter-notebook will open in (localhost):6565

## Citation
```
@article{guo2021sample,
  title={Sample and Computation Redistribution for Efficient Face Detection},
  author={Guo, Jia and Deng, Jiankang and Lattas, Alexandros and Zafeiriou, Stefanos},
  journal={arXiv preprint arXiv:2105.04714},
  year={2021}
}
```