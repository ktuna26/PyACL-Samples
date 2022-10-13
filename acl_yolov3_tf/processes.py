import acl, time, cv2
import sys
sys.path.append('../acllite')
from acllite_model import AclLiteModel
from acllite_resource import AclLiteResource
import acl, cv2, struct, time
import numpy as np
from PIL import Image, ImageDraw
import os




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

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0)]

def preprocessing(img,model_desc):
    #model_input_height, model_input_width ,_ ,_ = get_sizes(model_desc)
    image_height, image_width = img.shape[:2]
    img_resized = letterbox_resize(img, 416, 416)[:, :, ::-1]
    # img_resized = self.resize_image(img, (self.model_input_width, self.model_input_height))[:, :, ::-1]
    # img_resized = (img_resized / 255).astype(np.float32).transpose([2, 0, 1])
    img_resized = np.ascontiguousarray(img_resized)
    print("img_resized shape", img_resized.shape)
    return img_resized

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

def letterbox_resize(img, new_width, new_height, interp=0):
    '''
    Letterbox resize. keep the original aspect ratio in the resized image.
    '''
    ori_height, ori_width = img.shape[:2]
    resize_ratio = min(new_width / ori_width, new_height / ori_height)
    resize_w = int(resize_ratio * ori_width)
    resize_h = int(resize_ratio * ori_height)

    # img = cv2.resize(img, (resize_w, resize_h), interpolation=interp)
    # img = cv2.resize(img, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
    img = cv2.resize(img, (resize_w, resize_h), interpolation=cv2.INTER_NEAREST)

    image_padded = np.full((new_height, new_width, 3), 128, np.uint8)

    dw = int((new_width - resize_w) / 2)
    dh = int((new_height - resize_h) / 2)

    image_padded[dh: resize_h + dh, dw: resize_w + dw, :] = img
    # return image_padded, resize_ratio, dw, dh
    return image_padded


def get_model_output_by_result_list(model, result_list, original_image, score_threshold, iou_threshold):
    '''
    get_model_output_by_result_list(result_list, original_image, 0.3, 0.45)
    '''
    pred_sbbox = result_list[0].reshape(-1, 85)
    pred_mbbox = result_list[1].reshape(-1, 85)
    pred_lbbox = result_list[2].reshape(-1, 85)
    pred_bbox = np.concatenate([pred_sbbox, \
                                pred_mbbox, \
                                pred_lbbox], axis=0)
    original_image_size = original_image.shape[:2]
    bboxes = postprocess_boxes(pred_bbox, original_image_size, acl.mdl.get_input_dims(model._model_desc, 0)[0]['dims'][1], score_threshold)
    bboxes = nms(bboxes, iou_threshold, method='nms')
    return bboxes


def bboxes_iou(boxes1, boxes2):

    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up       = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down    = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area    = inter_section[..., 0] * inter_section[..., 1]
    union_area    = boxes1_area + boxes2_area - inter_area
    ious          = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return ious

def nms(bboxes, iou_threshold, sigma=0.3, method='nms'):
    """
    :param bboxes: (xmin, ymin, xmax, ymax, score, class)
    Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
          https://github.com/bharatsingh430/soft-nms
    """
    classes_in_img = list(set(bboxes[:, 5]))
    best_bboxes = []

    for cls in classes_in_img:
        cls_mask = (bboxes[:, 5] == cls)
        cls_bboxes = bboxes[cls_mask]

        while len(cls_bboxes) > 0:
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
            iou = bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            weight = np.ones((len(iou),), dtype=np.float32)

            assert method in ['nms', 'soft-nms']

            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0

            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))

            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > 0.
            cls_bboxes = cls_bboxes[score_mask]

    return best_bboxes
def postprocess_boxes(pred_bbox, org_img_shape, input_size, score_threshold):

    valid_scale=[0, np.inf]
    pred_bbox = np.array(pred_bbox)

    pred_xywh = pred_bbox[:, 0:4]
    pred_conf = pred_bbox[:, 4]
    pred_prob = pred_bbox[:, 5:]

    # # (1) (x, y, w, h) --> (xmin, ymin, xmax, ymax)
    pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)
    # # (2) (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
    org_h, org_w = org_img_shape
    resize_ratio = min(input_size / org_w, input_size / org_h)

    dw = (input_size - resize_ratio * org_w) / 2
    dh = (input_size - resize_ratio * org_h) / 2

    pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
    pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

    # # (3) clip some boxes those are out of range
    pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
    invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
    pred_coor[invalid_mask] = 0

    # # (4) discard some invalid boxes
    bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
    scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

    # # (5) discard some boxes with low scores
    classes = np.argmax(pred_prob, axis=-1)
    scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
    score_mask = scores > score_threshold
    mask = np.logical_and(scale_mask, score_mask)
    coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

    return np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)

