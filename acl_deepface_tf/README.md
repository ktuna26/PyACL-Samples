# Tensorflow Deepface (ArcFace)
Please open the jupyter notebook for a quick demo.

## Original Network Link

https://github.com/serengil/deepface

## Pre-trained Model Link:

Download the h5 file according to the instructions in the original repo.

## Convert model To Ascend om file

The h5 file should be first converted to ONNX format (opset 11) and then be converted to OM format.

```bash
cd ./model
atc --model=arcface.onnx \
    --framework=5 \
    --input_shape="your_model_input_name:1,112,112,3" \
    --output=./arcface \
    --soc_version={Ascend310, Ascend910A} \
    --out_nodes="embedding:0"
```