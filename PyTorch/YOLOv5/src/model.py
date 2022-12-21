"""
Copyright 2022 Huawei Technologies Co., Ltd

CREATED:  2022-10-04 13:12:13
MODIFIED: 2022-12-20 08:48:45
"""

# -*- coding:utf-8 -*-
import acl
import cv2


def get_sizes(model_desc):
    input_size = acl.mdl.get_num_inputs(model_desc)
    output_size = acl.mdl.get_num_outputs(model_desc)
    print("model input size", input_size)

    for i in range(input_size):
        print("input ", i)
        print("model input dims", acl.mdl.get_input_dims(model_desc, i))
        print("model input datatype", acl.mdl.get_input_data_type(model_desc, i))
        model_input_height, model_input_width = (i for i in acl.mdl.get_input_dims(model_desc, i)[0]['dims'][1:3])
        # model_input_element_number = acl.mdl.get_input_dims(model_desc, i)[0]['dimCount']
    print("=" * 95)
    print("model output size", output_size)

    for i in range(output_size):
        print("output ", i)
        print("model output dims", acl.mdl.get_output_dims(model_desc, i))
        print("model output datatype", acl.mdl.get_output_data_type(model_desc, i))
        model_output_height, model_output_width = acl.mdl.get_output_dims(model_desc, i)[0]['dims'][1:]
        # model_output_element_number = acl.mdl.get_output_dims(model_desc, i)[0]['dimCount']
    print("=" * 95)
    print("[Model] class Model init resource stage success")

    return model_input_height,model_input_width,model_output_height,model_output_width


def preprocessing(org_img,model_desc): # 1) pre-processing stage
    model_input_height, model_input_width, \
    model_output_height,model_output_width = get_sizes(model_desc)
    
    img_resized = letterbox(org_img[:, :, ::-1], (model_input_width, 
                    model_input_height))[0] # bgr to rgb (color space change) & resize
    print("[PreProc] img_resized shape", img_resized.shape)

    return img_resized, model_input_height, \
        model_input_width, model_output_height,model_output_width


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    
    return img, ratio, (dw, dh)