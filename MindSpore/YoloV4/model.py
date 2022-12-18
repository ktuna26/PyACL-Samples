"""
Copyright 2021 Huawei Technologies Co., Ltd

CREATED:  2022-11-23 13:12:13
MODIFIED: 2022-11-23 10:48:45
"""

# -*- coding:utf-8 -*-
import numpy as np
import acl
import cv2 as cv
from PIL import Image

# CONSTANTS
class_num = 80
stride_list = [32, 16, 8]
anchors_3 = np.array([[12, 16], [19, 36], [40, 28]]) / stride_list[2]
anchors_2 = np.array([[36, 75], [76, 55], [72, 146]]) / stride_list[1]
anchors_1 = np.array([[142, 110], [192, 243], [459, 401]]) / stride_list[0]
anchor_list = [anchors_1, anchors_2, anchors_3]
iou_threshold = 0.9

labels =["person",  "bicycle", "car", "motorbike", "aeroplane",
        "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench","bird", 
        "cat", "dog", "horse", "sheep", "cow", 
        "elephant",  "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag",  "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball",  "kite", "baseball bat",
        "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
        "wine glass", "cup", "fork", "knife", "spoon",
        "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
        "pizza", "donut", "cake", "chair", "sofa", "potted plant", "bed", "dining table",
        "toilet", "TV monitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush"]


def get_sizes(model_desc, returnType):
    input_size = acl.mdl.get_num_inputs(model_desc)
    output_size = acl.mdl.get_num_outputs(model_desc)
    if returnType:
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
    else:
        for i in range(input_size):
            model_input_height, model_input_width = acl.mdl.get_input_dims(model_desc, i)[0]['dims'][2:]
        return model_input_height, model_input_width

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


def overlap(x1, x2, x3, x4):
    left = max(x1, x3)
    right = min(x2, x4)
    return right - left

def cal_iou(box, truth):
    w = overlap(box[0], box[2], truth[0], truth[2])
    h = overlap(box[1], box[3], truth[1], truth[3])
    if w <= 0 or h <= 0:
        return 0
    inter_area = w * h
    union_area = (box[2] - box[0]) * (box[3] - box[1]) + (truth[2] - truth[0]) * (truth[3] - truth[1]) - inter_area
    return inter_area * 1.0 / union_area

def apply_nms(all_boxes, thres):
    res = []
    for cls in range(class_num):
        cls_bboxes = all_boxes[cls]
        sorted_boxes = sorted(cls_bboxes, key=lambda d: d[5])[::-1]
        p = dict()
        for i in range(len(sorted_boxes)):
            if i in p:
                continue
            truth = sorted_boxes[i]
            for j in range(i + 1, len(sorted_boxes)):
                if j in p:
                    continue
                box = sorted_boxes[j]
                iou = cal_iou(box, truth)
                if iou >= thres:
                    p[j] = 1
        for i in range(len(sorted_boxes)):
            if i not in p:
                res.append(sorted_boxes[i])
    return res


def decode_bbox(conv_output, anchors, img_w, img_h, x_scale, y_scale, shift_x_ratio, shift_y_ratio):
    print('conv_output.shape', conv_output.shape)
    _, h, w, _, _ = conv_output.shape 
    print(h,w)
    conv_output = conv_output.transpose(0, 1, 2, 3, -1)
    print('conv out-shape',conv_output.shape)
    pred = conv_output.reshape((h * w, 3, 5 + class_num))
    bbox = np.zeros((h * w, 3, 4))
    bbox[..., 0] = np.maximum((pred[..., 0] - pred[..., 2] / 2.0 - shift_x_ratio) * x_scale * img_w, 0)  # x_min
    bbox[..., 1] = np.maximum((pred[..., 1] - pred[..., 3] / 2.0 - shift_y_ratio) * y_scale * img_h, 0)  # y_min
    bbox[..., 2] = np.minimum((pred[..., 0] + pred[..., 2] / 2.0 - shift_x_ratio) * x_scale * img_w, img_w)  # x_max
    bbox[..., 3] = np.minimum((pred[..., 1] + pred[..., 3] / 2.0 - shift_y_ratio) * y_scale * img_h, img_h)  # y_max
    pred[..., :4] = bbox
    pred = pred.reshape((-1, 5 + class_num))
    print('pred-reshaped',pred.shape)
    pred[:, 4] = pred[:, 4] * pred[:, 5:].max(1)
    pred[:, 5] = np.argmax(pred[:, 5:], axis=-1)    
    pred = pred[pred[:, 4] >= 0.65]
    print('pred[:, 5]', pred[:, 5])
    print('pred[:, 5] shape', pred[:, 5].shape)

    all_boxes = [[] for ix in range(class_num)]
    for ix in range(pred.shape[0]):
        box = [int(pred[ix, iy]) for iy in range(4)]
        box.append(int(pred[ix, 5]))
        box.append(pred[ix, 4])
        all_boxes[box[4] - 1].append(box)
    return all_boxes


def convert_labels(label_list):
    if isinstance(label_list, np.ndarray):
        label_list = label_list.tolist()
        label_names = [labels[int(index)] for index in label_list]
    return label_names

def post_process(infer_output, origin_img, model_desc):
    MODEL_HEIGHT, MODEL_WIDTH = get_sizes(model_desc, False)
    print("post process")
    result_return = dict()
    img_h = origin_img.size[1]
    img_w = origin_img.size[0]
    scale = min(float(MODEL_WIDTH) / float(img_w), float(MODEL_HEIGHT) / float(img_h))
    new_w = int(img_w * scale)
    new_h = int(img_h * scale)
    shift_x_ratio = (MODEL_WIDTH - new_w) / 2.0 / MODEL_WIDTH
    shift_y_ratio = (MODEL_HEIGHT - new_h) / 2.0 / MODEL_HEIGHT
    class_number = len(labels)
    num_channel = 3 * (class_number + 5)
    x_scale = MODEL_WIDTH / float(new_w)
    y_scale = MODEL_HEIGHT / float(new_h)
    all_boxes = [[] for ix in range(class_number)]
    for ix in range(3):    
        pred = infer_output[ix]
        print('pred.shape', pred.shape)
        anchors = anchor_list[ix]
        boxes = decode_bbox(pred, anchors, img_w, img_h, x_scale, y_scale, shift_x_ratio, shift_y_ratio)
        #print(boxes)
        all_boxes = [all_boxes[iy] + boxes[iy] for iy in range(class_number)]

    res = apply_nms(all_boxes, iou_threshold)
    if not res:
        result_return['detection_classes'] = []
        result_return['detection_boxes'] = []
        result_return['detection_scores'] = []
        return result_return
    else:
        new_res = np.array(res)
        picked_boxes = new_res[:, 0:4]
        picked_boxes = picked_boxes[:, [1, 0, 3, 2]]
        picked_classes = convert_labels(new_res[:, 4])
        picked_score = new_res[:, 5]
        result_return['detection_classes'] = picked_classes
        result_return['detection_boxes'] = picked_boxes.tolist()
        result_return['detection_scores'] = picked_score.tolist()
        return result_return