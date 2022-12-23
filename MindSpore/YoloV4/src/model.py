"""
Copyright 2021 Huawei Technologies Co., Ltd

CREATED:  2022-23-11 13:12:13
MODIFIED: 2022-22-12 11:18:45
"""

# -*- coding:utf-8 -*-
import numpy as np
import acl
import cv2 as cv
from PIL import Image
from src.postprocess import get_sizes


# CONSTANTS
class_num = 80
stride_list = [32, 16, 8]
anchors_3 = np.array([[12, 16], [19, 36], [40, 28]]) / stride_list[2]
anchors_2 = np.array([[36, 75], [76, 55], [72, 146]]) / stride_list[1]
anchors_1 = np.array([[142, 110], [192, 243], [459, 401]]) / stride_list[0]
anchor_list = [anchors_1, anchors_2, anchors_3]
iou_threshold = 0.9

with open("./data/coco.names") as fd:
    coco_labels = fd.readlines()

labels = [i[:-1] for i in coco_labels][1:]


def preprocess(img_path, model_desc):
    MODEL_HEIGHT, MODEL_WIDTH = get_sizes(model_desc, True)
    image = Image.open(img_path)
    img_h = image.size[1]
    img_w = image.size[0]
    net_h = MODEL_HEIGHT
    net_w = MODEL_WIDTH

    scale = min(float(net_w) / float(img_w), float(net_h) / float(img_h))
    new_w = int(img_w * scale)
    new_h = int(img_h * scale)

    shift_x = (net_w - new_w) // 2
    shift_y = (net_h - new_h) // 2
    shift_x_ratio = (net_w - new_w) / 2.0 / net_w
    shift_y_ratio = (net_h - new_h) / 2.0 / net_h

    image_ = image.resize((new_w, new_h))
    new_image = np.zeros((net_h, net_w, 3), np.uint8)
    new_image[shift_y: new_h + shift_y, shift_x: new_w + shift_x, :] = np.array(image_)
    new_image = new_image.astype(np.float32)
    new_image = new_image / 255
    print('new_image.shape', new_image.shape)
    new_image = new_image.transpose(2, 0, 1).copy()
    return new_image, image
