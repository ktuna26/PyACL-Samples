# Tensorflow Deepface 
Please open the jupyter notebook for a quick demo.

## Original Network Link

https://github.com/serengil/deepface

## Pre-trained Model Link:

### H5 -> ONNX
Use this step to convert h5 to onnx

We recomend to use python virtual environment for H5->ONNX conversion.

- Example for python virtual environment:
```bash
python -m venv ENV_NAME

source ENV_NAME/bin/activate
```

- Change directiory to export folder.
- Install necessary python packages using requirements.txt file:

```bash
pip install -r requirements.txt
```

Use the onnx_converter.py script to convert H5 file to ONNX file.


```bash
python onnx_converter.py --model [MODEL_NAME] --output [OUTPUT_NAME]
```
- MODEL_NAME = arcface, vggface, facenet, deepface, openface or deepid
- OUTPUT_NAME = filename of converted onnx file.

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