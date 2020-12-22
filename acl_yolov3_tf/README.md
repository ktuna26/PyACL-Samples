# Tensorflow YoloV3 Object Detection
Please open the jupyter notebook for a quick demo.

## Original Network Link

https://github.com/YunYang1994/tensorflow-yolov3

## Pre-trained Model Link:

Download the frozen PB file of tensorflow yolov3 from this link,
- https://www.huaweicloud.com/intl/en-us/ascend/resources/modelzoo/Models/5c3ec43f66ba455a9992fff905c6d687
- Upload the pb file to `model` directory

## Convert model To Ascend om file

```bash
cd ./model
atc --model=yolov3_tf.pb \
    --framework=3 \
    --input_shape="input/input_data:1,416,416,3" \
    --output=./yolov3_coco_tf_rgb888 \
    --insert_op_conf=./aipp_yolov3_tf.cfg \
    --soc_version=Ascend310 \
    --out_nodes="pred_sbbox/concat_2:0;pred_mbbox/concat_2:0;pred_lbbox/concat_2:0"
```