# Object Detection Model

## Original Network Link

https://github.com/ChenYingpeng/caffe-yolov3

## Pre-trained Model Link:

### Caffe

https://obs-model-ascend.obs.cn-east-2.myhuaweicloud.com/yolov3/yolov3.caffemodel

Download the model weight file ``yolov3.caffemodel`` from this link or simply `wget` to ./model dir. 
Note that if you are developing applications for Atlas 500, you will need to make the model conversion in a separate development environment.

## Convert model To Ascend om file

### Caffe
```bash
cd ./model
atc --model=./yolov3.prototxt \
    --weight=./yolov3.caffemodel \
    --framework=0 \
    --output=./yolov3_caffe_416_no_csc \
    --soc_version=Ascend310 \
    --insert_op_conf=./aipp_yolov3_416_no_csc.cfg
```