# PyTorch CRAFT: Character-Region Awareness For Text detection
Please open the `jupyter-notebook` for a quick demo | [Pretrained Model](https://onebox.huawei.com/p/a79f35add575531ee601c5843abead7c) | [Paper](https://arxiv.org/abs/1904.01941) | [Original Github Repository](https://github.com/clovaai/CRAFT-pytorch)

## Overview
`PyTorch` implementation for **CRAFT** text detector that effectively detect text area by exploring each character region and affinity between characters. The bounding box of texts are obtained by simply finding minimum bounding rectangles on binary map after thresholding character region and affinity scores.

<img width="1000" alt="teaser" src="./figures/craft_example.gif">

## Getting started
Install dependencies;
- opencv-python>=3.4.2
- scikit-image>=0.14.2

```
pip install -r requirements.txt
```
And then download the `PT` file of from the link.

### PT model -> ONNX format -> Ascend om format
#### PT -> ONNX
Use in the original repository  the `model/onnx_export.py` the script to convert `PT` file to `ONNX ` file.

#### ONNX -> OM
And then use in the same directory atc tool to convert `ONNX ` file to `OM` file as as follows.
```bash
atc --model=craft.onnx \
    --framework=5 \
    --output=craft \
    --soc_version=Ascend310 \
    --precision_mode=allow_fp32_to_fp16
```

Finaly, open `jupyter-notebook` and run the code for demo

## Citation
```
@inproceedings{baek2019character,
  title={Character Region Awareness for Text Detection},
  author={Baek, Youngmin and Lee, Bado and Han, Dongyoon and Yun, Sangdoo and Lee, Hwalsuk},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={9365--9374},
  year={2019}
}
```