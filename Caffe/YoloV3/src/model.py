"""
Copyright 2021 Huawei Technologies Co., Ltd

CREATED:  2022-10-04 13:12:13
MODIFIED: 2022-21-12 09:48:45
"""

# -*- coding:utf-8 -*-
import numpy as np
import acl
import cv2
import os


with open("./data/coco.names") as fd:
    coco_labels = fd.readlines()

labels = [i[:-1] for i in coco_labels][1:]

MODEL_WIDTH = 416
MODEL_HEIGHT = 416

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0)]


def construct_image_info():
    """construct image info"""
    image_info = np.array([MODEL_WIDTH, MODEL_HEIGHT, 
                           MODEL_WIDTH, MODEL_HEIGHT], 
                           dtype = np.float32) 
    return image_info

def preprocessing(img,model_desc,model_name ="yolov3"):
    if model_name == "yolov3":
        model_width = 416
        model_height = 416
    elif model_name == "yolov4":
        model_width = 608
        model_height = 608
    else:
        raise TypeError('model name parameter is wrong!')

    img = cv2.resize(img, (model_width, model_height), interpolation = cv2.INTER_AREA)
    image_height, image_width = img.shape[:2]
    img_resized = letterbox_resize(img, model_width, model_height)[:, :, ::-1]
    img_resized = np.ascontiguousarray(img_resized)

    return img_resized

def letterbox_resize(img, new_width, new_height, interp=0):
    '''
    Letterbox resize. keep the original aspect ratio in the resized image.
    '''
    ori_height, ori_width = img.shape[:2]
    resize_ratio = min(new_width / ori_width, new_height / ori_height)
    resize_w = int(resize_ratio * ori_width)
    resize_h = int(resize_ratio * ori_height)

    img = cv2.resize(img, (resize_w, resize_h), interpolation=cv2.INTER_NEAREST)

    image_padded = np.full((new_height, new_width, 3), 128, np.uint8)

    dw = int((new_width - resize_w) / 2)
    dh = int((new_height - resize_h) / 2)

    image_padded[dh: resize_h + dh, dw: resize_w + dw, :] = img
    return image_padded

def get_sizes(model_desc):
    input_size = acl.mdl.get_num_inputs(model_desc)
    output_size = acl.mdl.get_num_outputs(model_desc)
    print("model input size", input_size)
    for i in range(input_size):
        print("input ", i)
        print("model input dims", acl.mdl.get_input_dims(model_desc, i))
        print("model input datatype", acl.mdl.get_input_data_type(model_desc, i))

    print("=" * 50)
    print("model output size", output_size)
    for i in range(output_size):
            print("output ", i)
            print("model output dims", acl.mdl.get_output_dims(model_desc, i))
            print("model output datatype", acl.mdl.get_output_data_type(model_desc, i))

            
    print("=" * 50)
    print("[Model] class Model init resource stage success")
