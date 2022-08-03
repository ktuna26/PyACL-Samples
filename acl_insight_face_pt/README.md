# InsightFace : SCRFD Face Detection
Please open the `jupyter-notebook` for a quick demo | [Pretrained Model](https://onebox.huawei.com/p/d2bb0e04156ba1c43091f7d8946eb293) | [Paper](https://arxiv.org/abs/2105.04714) | [Original Github Repository](https://github.com/deepinsight/insightface/tree/master/detection/scrfd)

<div align="left">
  <img src="./figures/insight_face.jpg" width="240" alt="prcurve"/>
</div>

## Overview
`SCRFD` is an efficient high accuracy face detection approach which 

<img src="./figures/scrfd_evelope.jpg" width="400" alt="prcurve"/>

## Getting started
Install dependencies;
- opencv-python>=3.4.2
-  Pillow
- numpy

```
pip install -r requirements.txt
```

And then download the `PT` file of from the link.

### PT model -> ONNX format -> Ascend om format
#### PT -> ONNX
Use inside the original repository tools folder  the `model/onnx_export.py` the script to convert `PT` file to `ONNX ` file.

- Download the original repository for convert pt model to onnx.

```bash
git clone https://github.com/deepinsight/insightface.git
```

- Copy files inside of the model directory to `insightface/detection/scrfd/tools` folder in the original repository.

```bash
cp acl_insight_face_pt/model/*  {path}/insightface/detection/scrfd/tools/ 
```

- Change directory to `scrfd/`, then run the `model/onnx_converter_env_setup.sh`
```bash
cd {path}/insightface/detection/scrfd/
chmod +x tools/onnx_converter_env_setup.sh
./tools/onnx_converter_env_setup.sh
```
- Run below python command to export `.onnx` model

```bash 
# Modify paths for you.
python3 onnx_export.py --config scrfd_34g.py \
                       --weights scrfd_34g.pth \
                       --input_img ../data/sample.jpg \
                       --simplify
```


#### ONNX -> OM
And then use in the same directory atc tool to convert `ONNX ` file to `OM` file as as follows.
```bash
atc --model=scrfd_34g.onnx \
    --framework=5 \
    --output=scrfd_34g \ 
    --soc_version=Ascend310 \
    --precision_mode=allow_fp32_to_fp16
```

Finaly, open `jupyter-notebook` and run the code for demo

## Citation
```
@article{guo2021sample,
  title={Sample and Computation Redistribution for Efficient Face Detection},
  author={Guo, Jia and Deng, Jiankang and Lattas, Alexandros and Zafeiriou, Stefanos},
  journal={arXiv preprint arXiv:2105.04714},
  year={2021}
}
```