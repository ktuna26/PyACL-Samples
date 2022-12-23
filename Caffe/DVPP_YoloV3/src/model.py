"""
Copyright 2021 Huawei Technologies Co., Ltd

CREATED:  2022-10-04 13:12:13
MODIFIED: 2022-21-12 09:28:45
"""

# -*- coding:utf-8 -*-
import numpy as np
import acl

with open("./data/coco.names") as fd:
    coco_labels = fd.readlines()

labels = [i[:-1] for i in coco_labels][1:]

MODEL_WIDTH = 416
MODEL_HEIGHT = 416

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