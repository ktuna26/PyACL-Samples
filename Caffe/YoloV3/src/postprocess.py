"""
Copyright 2021 Huawei Technologies Co., Ltd

CREATED:  2022-10-04 13:12:13
MODIFIED: 2022-21-12 09:48:45
"""

# -*- coding:utf-8 -*-
import cv2
import os

with open("./data/coco.names") as fd:
    coco_labels = fd.readlines()

labels = [i[:-1] for i in coco_labels][1:]

MODEL_WIDTH = 416
MODEL_HEIGHT = 416

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0)]

def post_process(infer_output, bgr_img, image_file,model_name="yolov3"):
    """postprocess"""
    print("post process")
    box_num = infer_output[1][0, 0]
    box_info = infer_output[0].flatten() 
   
    scalex = bgr_img.shape[1] / MODEL_WIDTH
    scaley = bgr_img.shape[0] / MODEL_HEIGHT

    
    if not os.path.exists('./out'):
        os.makedirs('./out')
    output_path = os.path.join("./out", os.path.basename(image_file))
    print("image file = ", image_file)
    
    for n in range(int(box_num)):
        ids = int(box_info[5 * int(box_num) + n]) 
        label = labels[ids] 
        score = box_info[4 * int(box_num)+n]
        top_left_x = box_info[0 * int(box_num) + n] * scalex
        top_left_y = box_info[1 * int(box_num) + n] * scaley
        bottom_right_x = box_info[2 * int(box_num) + n] * scalex
        bottom_right_y = box_info[3 * int(box_num) + n] * scaley
        print(" % s: class % d, box % d % d % d % d, score % f" % (
            label, ids, top_left_x, top_left_y, 
            bottom_right_x, bottom_right_y, score))
        cv2.rectangle(bgr_img, (int(top_left_x), int(top_left_y)), 
                (int(bottom_right_x), int(bottom_right_y)), colors[n % 6])
        p3 = (max(int(top_left_x), 15), max(int(top_left_y), 15))
        cv2.putText(bgr_img, label, p3, cv2.FONT_ITALIC, 0.6, colors[n % 6], 1)

    output_file = os.path.join("./out", "out_" + os.path.basename(image_file))
    print("output:%s" % output_file)
    cv2.imwrite(output_file, bgr_img)
    print("success!")
    return bgr_img