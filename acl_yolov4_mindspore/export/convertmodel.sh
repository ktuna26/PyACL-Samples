#!/bin/bash

echo "[CKPT->AIR] Model Conversion Started"
python3 export.py --ckpt_file yolov4_ascend_v180_coco2017_official_cv_acc44.ckpt --file_name yolov4_latest --file_format AIR --keep_detect True
echo "[CKPT->AIR] Model Conversion Ended"
echo "[AIR->OM]"
echo "atc --output=../model/yolov4_bs1 --soc_version=$device --framework=1 --model=./yolov4_latest.air"
