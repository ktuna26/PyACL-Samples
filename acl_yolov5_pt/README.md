# PyTorch YoloV5 Object Detection
Please open the jupyter notebook for a quick demo. This sample uses  **yolov5s**.  
Addtionally, the code supported PT yolov3 (https://www.hiascend.com/zh/software/modelzoo/detail/1/36ea401e0d844f549da2693c6289ad89)

## Original Network Link

https://github.com/ultralytics/yolov5

## Pre-trained Model Link:

Download the PT file of from this link,
- https://github.com/ultralytics/yolov5/releases/tag/v2.0
- Upload the pt file to `model` directory

## PT model -> ONNX format -> Ascend om format
### PT -> ONNX
Use the onnx_exporter/export.py script in this repository to convert PT file to ONNX file.  
Use this step to convert  **yolov3.pt**  to  **yolov3_sim.onnx**. 

```
python onnx_exporter/export.py --weights model/yolov3.pt --img-size 416 --batch-size 1 --simplify
```

### Remove a few operators in the ONNX file
The  **Slice** and  **Transpose** operators will slow down the model inference significantly. Use ./model/modify_yolov5.py script in this repo to remove the impact of these operators.  
This step is **NOT** needed for yolov3.

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
The **\-\-out_nodes** names can vary, adjust the parameter accordingly.

```
atc --model=yolov3_sim.onnx \
    --framework=5 \
    --output=yolov3_bs1_aipp \
    --input_format=NCHW \
    --soc_version=Ascend310 \
    --input_shape="images:1,3,416,416" \
    --out_nodes="Transpose_274:0;Transpose_258:0;Transpose_242:0" \
    --insert_op_conf=aipp_yolov3.cfg\
    --output_type=FP16
```