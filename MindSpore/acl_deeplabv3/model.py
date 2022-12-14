"""
Copyright 2021 Huawei Technologies Co., Ltd

CREATED:  2022-11-23 13:12:13
MODIFIED: 2022-11-23 10:48:45
"""

# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np
import os
import sys
import acl
path = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.join(path, "../acllite"))

import acllite_utils as utils
import draw_predict


def get_sizes(model_desc):
    input_size = acl.mdl.get_num_inputs(model_desc)
    output_size = acl.mdl.get_num_outputs(model_desc)
    print("model input size", input_size)
    for i in range(input_size):
        print("input ", i)
        print("model input dims", acl.mdl.get_input_dims(model_desc, i))
        print("model input datatype", acl.mdl.get_input_data_type(model_desc, i))
        model_input_height, model_input_width = acl.mdl.get_input_dims(model_desc, i)[0]['dims'][2:]
    print("=" * 50)
    print("model output size", output_size)
    for i in range(output_size):
            print("output ", i)
            print("model output dims", acl.mdl.get_output_dims(model_desc, i))
            print("model output datatype", acl.mdl.get_output_data_type(model_desc, i))
            model_output_height, model_output_width = acl.mdl.get_output_dims(model_desc, i)[0]['dims'][1:3]
            
    print("=" * 50)
    print("[Model] class Model init resource stage success")
    return model_input_height,model_input_width

@utils.display_time
def preprocess(picPath,model_desc):
    model_input_height,model_input_width = get_sizes(model_desc)
    bgr_img_ = cv.imread(picPath).astype(np.uint8)

    img = cv.resize(bgr_img_, (model_input_width, model_input_height))
    img=img.astype(np.float32, copy=False)

    img[:, :, 0] -= 104
    img[:, :, 0] = img[:, :, 0] / 57.375
    img[:, :, 1] -= 117
    img[:, :, 1] = img[:, :, 1] / 57.120
    img[:, :, 2] -= 123
    img[:, :, 2] = img[:, :, 2] / 58.395
    img = img.transpose([2, 0, 1]).copy()
    return img


@utils.display_time
def postprocess(result_list, pic, output_dir):
    # +++++++++TEST+++++++++++++++++
    #print('BEFORE SIGMOID',result_list[0])
    draw_predict.draw_label(pic, result_list[0].squeeze(), output_dir)
    