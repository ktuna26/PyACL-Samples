"""
Copyright 2022 Huawei Technologies Co., Ltd

CREATED:  2022-10-04 13:12:13
MODIFIED: 2022-12-13 10:48:45
"""

# -*- coding:utf-8 -*-
import acl
import numpy as np

labels = ["person",
        "bicycle", "car", "motorbike", "aeroplane",
        "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
        "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
        "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
        "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
        "pizza", "donut", "cake", "chair", "sofa", "potted plant", "bed", "dining table",
        "toilet", "TV monitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush"]

MODEL_WIDTH = 416
MODEL_HEIGHT = 416


def get_model_info(model):
    i = 0
    o = 0
    print(f'=================================\n\t\033[1mInput Dimensions\033[0m\n=================================')
    while i>=0:
        try:
            input_dims = acl.mdl.get_input_dims(model._model_desc, i)

            print(f"\033[32mName\033[0m: {input_dims[0]['name']}\n\033[36mDimensions\033[0m: {input_dims[0]['dims']}\n---------------------------------")
            i += 1
            acl.mdl.get_input_dims(model._model_desc, i)[0]['dims']
        except: i = -1
    print('='*33)
    print(f'\n\n=================================\n\t\033[1mOutput Dimensions\033[0m\n=================================')
    while o>=0:
        try:
            output_dims = acl.mdl.get_output_dims(model._model_desc, o)
            print(f"\033[32mName\033[0m: {output_dims[0]['name']}\n\033[36mDimensions\033[0m: {output_dims[0]['dims']}\n---------------------------------")
            o += 1
            acl.mdl.get_output_dims(model._model_desc, o)[0]['dims']
        except: o = -1
    print('='*33)
    
    
def construct_image_info():
    """construct image info"""
    image_info = np.array([MODEL_WIDTH, MODEL_HEIGHT, 
                           MODEL_WIDTH, MODEL_HEIGHT], 
                           dtype = np.float32) 
    return image_info

def pre_process(image, dvpp):
    """preprocess"""
    image_input = image.copy_to_dvpp()
    yuv_image = dvpp.jpegd(image_input)
    print("decode jpeg end")
    resized_image = dvpp.crop_and_paste(yuv_image, image.width, image.height, MODEL_WIDTH, MODEL_HEIGHT)
    print("resize yuv end")
    return resized_image

def post_processing(result_list):
    all_detections = []
    for j in range(result_list[1][0][0]):
        detection_object = {}
        detection_object['x1'] = result_list[0][0][0 * result_list[1][0][0] + j]
        detection_object['y1'] = result_list[0][0][1 * result_list[1][0][0] + j]
        detection_object['x2'] = result_list[0][0][2 * result_list[1][0][0] + j]
        detection_object['y2'] = result_list[0][0][3 * result_list[1][0][0] + j]
        detection_object['detection_scores']  = float(result_list[0][0][4 * result_list[1][0][0] + j])
        detection_object['detection_classes'] = result_list[0][0][5 * result_list[1][0][0] + j]
        detection_object['class_label'] = labels[int(result_list[0][0][5 * result_list[1][0][0] + j])]
        all_detections.append(detection_object)
    return all_detections