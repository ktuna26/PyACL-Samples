# PyTorch YoloV5 Object Detection
Please open the jupyter notebook for a quick demo. This sample uses  **yolov5s**. 

## Original Network Link

https://github.com/ultralytics/yolov5

## Pre-trained Model Link:

Download the PT file of from this link,
- https://github.com/ultralytics/yolov5/releases/tag/v4.0
- Upload the pt file to `model` directory

## PT model -> ONNX format -> Ascend om format
### PT -> ONNX
Use the onnx_exporter/export.py script in this repository to convert PT file to ONNX file. 
**Note :** CANN 20.2 support ONNX operators version 11.

### Remove a few operators in the ONNX file
The  **Slice** and  **Transpose** operators will slow down the model inference significantly. Use ./model/modify_yolov5.py script in this repo to remove the impact of these operators.

### ONNX -> OM
```bash
cd ./model
atc --model=modify_yolov5s.onnx \
    --framework=5 \
    --output=modify_yolov5s_out \
    --soc_version=Ascend310 \
    --input_shape="images:1,12,320,320" \
    --out_nodes="Reshape_259:0;Reshape_275:0;Reshape_291:0"
```