# Object Detection Model

## Original Network Link

https://github.com/ChenYingpeng/caffe-yolov3

## Pre-trained Model Link:

### Caffe

https://obs-model-ascend.obs.cn-east-2.myhuaweicloud.com/yolov3/yolov3.caffemodel

Download the model weight file ``yolov3.caffemodel`` from this link or simply wget

### Tensorflow

https://github.com/wizyoung/YOLOv3_TensorFlow/releases/

Download the checkpoint file, and convert it to pb file
Instructions: https://bbs.huaweicloud.com/forum/thread-68475-1-1.html

## Dependency
Please refer to the Developer Manual to customize the prototxt file and aipp config file

### Modify the prototxt file
**If the model is Caffe, you must modify the prototxt file by referring to the following link. Otherwise, an error will occur when running the sample**

https://support.huaweicloud.com/ti-atc-A800_3000_3010/altasatc_16_024.html

### Modify the aipp configuration file
https://support.huaweicloud.com/ti-atc-A800_3000_3010/altasatc_16_007.html

## Convert model To Ascend om file

### Caffe
```bash
cd yolov3
atc --model=./yolov3.prototxt \
    --weight=./yolov3.caffemodel \
    --framework=0 \
    --output=./yolov3_aipp \
    --soc_version=Ascend310 \
    --insert_op_conf=./aipp_yolov3.cfg
```

### Tensorflow
```bash
atc --model=./yolov3.pb \
    --framework=3 \
    --output=./yolov3_aipp \
    --input_shape="Placeholder:1,416,416,3" \
    --insert_op_conf=./aipp_yolov3.cfg \
    --soc_version=Ascend310
```

## Model replacement

Replace the YoloV3 model with other input specifications, we need to modify the configuration as follow:

### Configure setup.config
./data/config/setup.config
```bash
model_width = xxx
model_height = xxx
```

### Configure aipp_yolov3.cfg
./data/model/yolov3/aipp_yolov3.cfg
```bash
src_image_size_w : xxx
src_image_size_h : xxx
```

### Configureyolov3.prototxt(Caffe Model)
```bash
input_shape {
  dim: 1
  dim: 3
  dim: xxx
  dim: xxx
}
```
xxx is the new model input weight and height

## Products that have been verified:

- Atlas 800 (Model 3000)
- Atlas 800 (Model 3010)
- Atlas 300 (Model 3010)