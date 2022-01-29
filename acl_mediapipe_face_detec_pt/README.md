# PyTorch Mediapipe Face Detection (BlazeFace)
Please open the `jupyter-notebook` for a quick demo!
BlazeFace is a fast, light-weight face detector from Google Research | [Read more](https://sites.google.com/view/perception-cv4arvr/blazeface) | [Paper on arXiv](https://arxiv.org/abs/1907.05047) | [Pretrained Model](https://github.com/google/mediapipe/blob/v0.7.12/mediapipe/models/face_detection_back.tflite)

<img alt="teaser" src="./figures/mediapipe_small.png">

## Overview
`PyTorch` implementation for **Mediapipe** face detector that effectively detect face area by exploring 6 keypoints (2x eyes, 2x ears, nose, mouth) for face landmarks.
The BlazePaper paper mentions that there are two versions of the model, one for the front-facing camera and one for the back-facing camera. This repo includes only the backend camera model

<img width="1000" alt="teaser" src="./figures/face_detection_android_gpu.gif">

## Getting started
Install dependencies;
- opencv-python-headless
- numpy
- Pillow

```
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