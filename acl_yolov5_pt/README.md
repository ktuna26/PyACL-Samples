# PyTorch YoloV5 Object Detection
Please open the jupyter notebook for a quick demo. This sample uses  **yolov5s**. 

## Original Network Link

https://github.com/ultralytics/yolov5

## Pre-trained Model Link:

Download the PT file of from this link,
- https://github.com/ultralytics/yolov5/releases/tag/v2.0
- Upload the pt file to `model` directory

## PT model -> ONNX format -> Ascend om format
### PT -> ONNX
Use the onnx_exporter/export.py script in this repository to convert PT file to ONNX file.

### Remove a few operators in the ONNX file
The  **Slice** and  **Transpose** operators will slow down the model inference significantly. Use ./model/modify_yolov5.py script in this repo to remove the impact of these operators.

### ONNX -> OM
```bash
cd ./model
atc --model=yolov5s_sim_t.onnx \
    --framework=5 \
    --output=yolov5s_sim_aipp \
    --input_format=NCHW \
    --log=error \
    --soc_version=Ascend310 \
    --input_shape="images:1,3,640,640" \
    --enable_small_channel=1 \
    --output_type=FP16 \
    --insert_op_conf=aipp.cfg
```