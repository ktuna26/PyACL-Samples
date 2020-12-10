# Object Detection Model

## Original Network Link

https://github.com/ChenYingpeng/caffe-yolov3

## Pre-trained Model Link:

### Caffe

https://obs-model-ascend.obs.cn-east-2.myhuaweicloud.com/yolov3/yolov3.caffemodel

Download the model weight file ``yolov3.caffemodel`` from this link or simply wget

## Convert model To Ascend om file

### Caffe
```bash
cd yolov3
atc --model=./yolov3.prototxt \
    --weight=./yolov3.caffemodel \
    --framework=0 \
    --output=./yolov3_caffe \
    --soc_version=Ascend310 \
    --insert_op_conf=./aipp_yolov3.cfg
```