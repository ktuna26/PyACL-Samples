"""
Copyright 2021 Huawei Technologies Co., Ltd

CREATED:  2022-10-04 13:12:13
MODIFIED: 2022-21-12 09:28:45
"""

# -*- coding:utf-8 -*-

with open("./data/coco.names") as fd:
    coco_labels = fd.readlines()

labels = [i[:-1] for i in coco_labels][1:]

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