# Tensorflow YoloV3 Object Detection
Please open the jupyter notebook for a quick demo.

## Original Network Link

https://github.com/YunYang1994/tensorflow-yolov3

## Pre-trained Model Link:

### Caffe

Comming soon.

## Convert model To Ascend om file

```bash
cd ./model
atc --model=yolov3_coco_tf.pb \
    --framework=3 \
    --input_shape="input/input_data:1,416,416,3" \
    --output=./yolov3_coco_tf_rgb888 \
    --insert_op_conf=./aipp_yolov3_tf.cfg \
    --soc_version=Ascend310 \
    --out_nodes="pred_sbbox/concat_2:0;pred_mbbox/concat_2:0;pred_lbbox/concat_2:0"
```