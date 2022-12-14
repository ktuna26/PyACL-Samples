"""
Copyright 2021 Huawei Technologies Co., Ltd

CREATED:  2022-11-23 13:12:13
MODIFIED: 2022-11-23 10:48:45
"""

# -*- coding:utf-8 -*-
import acl
import numpy as np
from PIL import Image
from collections import defaultdict


def preprocess(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((640, 640), 0)
    img = np.array(img, dtype=np.float32)
    img = img / 255.
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)
    image = np.expand_dims(img, 0)
    data = np.concatenate((image[..., ::2, ::2],
                           image[..., 1::2, ::2],
                           image[..., ::2, 1::2],
                           image[..., 1::2, 1::2]),
                          axis=1)
    return data


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
    
class DetectionEngine:
    """Detection engine."""

    def __init__(self, img_shape, threshold):
        self.image_shape = img_shape
        self.ignore_threshold = threshold
        self.labels = ["person",
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
        "scissors", "teddy bear", "hair drier", "toothbrush"] # args_detection.labels
        self.num_classes = len(self.labels)
        self.results = {}
        self.file_path = ''
        self.save_prefix = '.' # args_detection.output_dir
        self.ann_file = None # args_detection.ann_file
        self._coco = None # COCO(self.ann_file)
        self._img_ids = None # list(sorted(self._coco.imgs.keys()))
        self.det_boxes = []
        self.nms_thresh = 0.45 # args_detection.eval_nms_thresh
        self.multi_label = True # args_detection.multi_label
        self.multi_label_thresh = 0.6 # args_detection.multi_label_thresh
        #self.coco_catids = self._coco.getCatIds()
        self.coco_catIds = np.arange(0,81) # args_detection.coco_ids

    def do_nms_for_results(self):
        """Get result boxes."""
        for img_id in self.results:
            for clsi in self.results[img_id]:
                dets = self.results[img_id][clsi]
                dets = np.array(dets)
                keep_index = self._diou_nms(dets, thresh=self.nms_thresh)

                keep_box = [{'image_id': int(img_id), 'category_id': int(clsi),
                             'bbox': list(dets[i][:4].astype(float)),
                             'score': dets[i][4].astype(float)} for i in keep_index]
                self.det_boxes.extend(keep_box)

    def _nms(self, predicts, threshold):
        """Calculate NMS."""
        # convert xywh -> xmin ymin xmax ymax
        x1 = predicts[:, 0]
        y1 = predicts[:, 1]
        x2 = x1 + predicts[:, 2]
        y2 = y1 + predicts[:, 3]
        scores = predicts[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        reserved_boxes = []
        while order.size > 0:
            i = order[0]
            reserved_boxes.append(i)
            max_x1 = np.maximum(x1[i], x1[order[1:]])
            max_y1 = np.maximum(y1[i], y1[order[1:]])
            min_x2 = np.minimum(x2[i], x2[order[1:]])
            min_y2 = np.minimum(y2[i], y2[order[1:]])

            intersect_w = np.maximum(0.0, min_x2 - max_x1 + 1)
            intersect_h = np.maximum(0.0, min_y2 - max_y1 + 1)
            intersect_area = intersect_w * intersect_h
            ovr = intersect_area / \
                (areas[i] + areas[order[1:]] - intersect_area)

            indexes = np.where(ovr <= threshold)[0]
            order = order[indexes + 1]
        return reserved_boxes

    def _diou_nms(self, dets, thresh=0.5):
        """
        convert xywh -> xmin ymin xmax ymax
        """
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = x1 + dets[:, 2]
        y2 = y1 + dets[:, 3]
        scores = dets[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            center_x1 = (x1[i] + x2[i]) / 2
            center_x2 = (x1[order[1:]] + x2[order[1:]]) / 2
            center_y1 = (y1[i] + y2[i]) / 2
            center_y2 = (y1[order[1:]] + y2[order[1:]]) / 2
            inter_diag = (center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2
            out_max_x = np.maximum(x2[i], x2[order[1:]])
            out_max_y = np.maximum(y2[i], y2[order[1:]])
            out_min_x = np.minimum(x1[i], x1[order[1:]])
            out_min_y = np.minimum(y1[i], y1[order[1:]])
            outer_diag = (out_max_x - out_min_x) ** 2 + (out_max_y - out_min_y) ** 2
            diou = ovr - inter_diag / outer_diag
            diou = np.clip(diou, -1, 1)
            inds = np.where(diou <= thresh)[0]
            order = order[inds + 1]
        return keep

    def write_result(self):
        """Save result to file."""
        import json
        t = datetime.datetime.now().strftime('_%Y_%m_%d_%H_%M_%S')
        try:
            self.file_path = self.save_prefix + '/predict' + t + '.json'
            f = open(self.file_path, 'w')
            json.dump(self.det_boxes, f)
        except IOError as e:
            raise RuntimeError("Unable to open json file to dump. What(): {}".format(str(e)))
        else:
            f.close()
            return self.file_path

    def get_eval_result(self):
        """Get eval result."""
        coco_gt = COCO(self.ann_file)
        coco_dt = coco_gt.loadRes(self.file_path)
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        rdct = Redirct()
        stdout = sys.stdout
        sys.stdout = rdct
        coco_eval.summarize()
        sys.stdout = stdout
        return rdct.content

    def detect(self, outputs, batch, image_shape, image_id):
        """Detect boxes."""
        # output [|32, 52, 52, 3, 85| ]
        for batch_id in range(batch):
            for out_item in outputs:
                # 52, 52, 3, 85
                out_item_single = out_item[batch_id, :]
                ori_w, ori_h = self.image_shape[:2]#[batch_id]
                img_id = 0 # int(image_id[batch_id])
                if img_id not in self.results:
                    self.results[img_id] = defaultdict(list)
                x = ori_w * out_item_single[..., 0].reshape(-1)
                y = ori_h * out_item_single[..., 1].reshape(-1)
                w = ori_w * out_item_single[..., 2].reshape(-1)
                h = ori_h * out_item_single[..., 3].reshape(-1)
                conf = out_item_single[..., 4:5]
                cls_emb = out_item_single[..., 5:]
                cls_argmax = np.expand_dims(np.argmax(cls_emb, axis=-1), axis=-1)
                x_top_left = x - w / 2.
                y_top_left = y - h / 2.
                cls_emb = cls_emb.reshape(-1, self.num_classes)
                if self.multi_label:
                    confidence = conf.reshape(-1, 1) * cls_emb
                    # create all False
                    flag = cls_emb > self.multi_label_thresh
                    flag = flag.nonzero()
                    for i, j in zip(*flag):
                        confi = confidence[i][j]
                        if confi < self.ignore_threshold:
                            continue
                        x_lefti, y_lefti = max(0, x_top_left[i]), max(0, y_top_left[i])
                        wi, hi = min(w[i], ori_w), min(h[i], ori_h)
                        # transform catId to match coco
                        coco_clsi = self.coco_catIds[j]
                        self.results[img_id][coco_clsi].append([x_lefti, y_lefti, wi, hi, confi])
                else:
                    cls_argmax = cls_argmax.reshape(-1)
                    # create all False
                    flag = np.random.random(cls_emb.shape) > sys.maxsize
                    for i in range(flag.shape[0]):
                        c = cls_argmax[i]
                        flag[i, c] = True
                    confidence = conf.reshape(-1) * cls_emb[flag]
                    for x_lefti, y_lefti, wi, hi, confi, clsi in zip(x_top_left, y_top_left,
                                                                     w, h, confidence, cls_argmax):
                        if confi < self.ignore_threshold:
                            continue
                        x_lefti, y_lefti = max(0, x_lefti), max(0, y_lefti)
                        wi, hi = min(wi, ori_w), min(hi, ori_h)
                        # transform catId to match coco
                        coco_clsi = self.coco_catids[clsi]
                        self.results[img_id][coco_clsi].append([x_lefti, y_lefti, wi, hi, confi])
