"""
Copyright 2021 Huawei Technologies Co., Ltd

CREATED:  2020-6-04 20:12:13
MODIFIED: 2021-11-02 23:48:45
"""

# -*- coding:utf-8 -*-
import acl
import numpy as np
import cv2
import time
from acl_util import check_ret
from constant import ACL_MEM_MALLOC_NORMAL_ONLY, \
                                    ACL_MEMCPY_HOST_TO_DEVICE, \
                                    ACL_MEMCPY_DEVICE_TO_HOST, \
                                    ACL_ERROR_NONE, NPY_BYTE
from postprocessing import get_model_output_by_index, letterbox, focus_process, \
                                               resize_image, detect, non_max_suppression, scale_coords

class Model(object):
    def __init__(self,
                 device_id,
                 model_path):
        self.device_id = device_id
        self.model_path = model_path    # string
        self.model_id = None            # pointer
        self.context = None             # pointer
        self.stream = None
        self.input_data = None
        self.output_data = None
        self.model_desc = None          # pointer when using
        self.input0_dataset_buffer = None
        self.input1_dataset_buffer = None
        self.input1_buffer = None
        self.model_input_width = None
        self.model_input_height = None
        self.input_dataset = None
        self.yolo_shapes= []
        self.element_number = None
        self.init_resource()
        

    def __del__(self):
        self._release_dataset()
        if self.model_id:
            ret = acl.mdl.unload(self.model_id)
            check_ret("acl.mdl.unload", ret)

        if self.model_desc:
            ret = acl.mdl.destroy_desc(self.model_desc)
            check_ret("acl.mdl.destroy_desc", ret)

        if self.input1_buffer:
            ret = acl.rt.free(self.input1_buffer)
            check_ret("acl.rt.free", ret)

        print("[Model] class Model release source success")
        
        if self.stream:
            acl.rt.destroy_stream(self.stream)

        if self.context:
            acl.rt.destroy_context(self.context)
        acl.rt.reset_device(self.device_id)
        acl.finalize()
        
        print("[ACL] class Sample release source success")

    def init_resource(self):
        print("[ACL] init resource stage:")
        acl.init()
        ret = acl.rt.set_device(self.device_id)
        check_ret("acl.rt.set_device", ret)

        self.context, ret = acl.rt.create_context(self.device_id)
        check_ret("acl.rt.create_context", ret)

        self.stream, ret = acl.rt.create_stream()
        check_ret("acl.rt.create_stream", ret)
        print("[ACL] init resource stage success")
        
        print("[Model] class Model init resource stage:")
        # context
        acl.rt.set_context(self.context)
        # load_model
        self.model_id, ret = acl.mdl.load_from_file(self.model_path)
        check_ret("acl.mdl.load_from_file", ret)
        self.model_desc = acl.mdl.create_desc()
        ret = acl.mdl.get_desc(self.model_desc, self.model_id)
        check_ret("acl.mdl.get_desc", ret)
        
        input_size = acl.mdl.get_num_inputs(self.model_desc)
        output_size = acl.mdl.get_num_outputs(self.model_desc)
        self._gen_output_dataset(output_size)
        print("model input size", input_size)
        for i in range(input_size):
            print("input ", i)
            print("model input dims", acl.mdl.get_input_dims(self.model_desc, i))
            print("model input datatype", acl.mdl.get_input_data_type(self.model_desc, i))
            self.model_input_height, self.model_input_width = (i for i in acl.mdl.get_input_dims(self.model_desc, i)[0]['dims'][1:3])
        print("=" * 50)
        print("model output size", output_size)
        for i in range(output_size):
            print("output ", i)
            print("model output dims", acl.mdl.get_output_dims(self.model_desc, i))
            print("model output datatype", acl.mdl.get_output_data_type(self.model_desc, i))
            self.yolo_shapes.append(acl.mdl.get_output_dims(self.model_desc, i)[0]['dims'])
            self.element_number = acl.mdl.get_output_dims(self.model_desc, i)[0]['dims'][-1]
        print("=" * 50)
        print("[Model] class Model init resource stage success")
   
    def transfer_img_to_device(self, img_resized):
        
        # BGR to RGB
        img_host_ptr, _ = acl.util.numpy_contiguous_to_ptr(img_resized)
        # print(img_host_ptr)
        img_buf_size = img_resized.itemsize * img_resized.size
        # print("img_buf_size", img_buf_size)
        img_dev_ptr, ret = acl.rt.malloc(img_buf_size, ACL_MEM_MALLOC_NORMAL_ONLY)
        check_ret("acl.rt.malloc", ret)
        ret = acl.rt.memcpy(img_dev_ptr, img_buf_size, img_host_ptr, img_buf_size, ACL_MEMCPY_HOST_TO_DEVICE)
        check_ret("acl.rt.memcpy", ret)
        
        return img_dev_ptr, img_buf_size
    
    def run1(self, img):
        self.img = letterbox(img, (self.model_input_width, self.model_input_height))[0]
        # self.img = self.img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x640x640
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        # image_np = np.array(self.img, dtype=np.float32)
        # image_np /= 255.0
        # image_np = image_np.astype(np.float16)
        image_np = self.img
        # image_np_expanded = np.expand_dims(image_np, axis=0)  # NCHW
        # Focus
        # img_numpy = focus_process(image_np_expanded)
        # img_numpy = image_np_expanded
        # print("image_np_expanded shape:", img_numpy.shape)
        # img_numpy = np.ascontiguousarray(img_numpy)
        
        img_dev_ptr, img_buf_size = self.transfer_img_to_device(image_np)
#         print("img_dev_ptr, img_buf_size: ", img_dev_ptr, img_buf_size)
        self._gen_input_dataset(img_dev_ptr, img_buf_size)

        t = time.time()
        self.forward()
        print("inference takes", time.time()-t)
        ret = acl.rt.free(img_dev_ptr)
        check_ret("acl.rt.free", ret)
        
        t = time.time()
        pred_sbbox = get_model_output_by_index(self.output_data, 0)
        pred_mbbox = get_model_output_by_index(self.output_data, 1)
        pred_lbbox = get_model_output_by_index(self.output_data, 2)
        feature_maps = [pred_sbbox, pred_mbbox, pred_lbbox]
        print("moving data takes", time.time()-t)
        # t = time.time()
        # print("self.yolo_shapes", self.yolo_shapes)
        for idx, feat, tgt_shape in zip(range(3), feature_maps, self.yolo_shapes):
            feature_maps[idx] = feat.reshape(tgt_shape)
        t = time.time()
        # print("feature_maps shape", feature_maps[0].shape)
        # print("self.element_number", self.element_number)
        res_tensor = detect(feature_maps, self.element_number)
        print("detect takes", time.time()-t)
        t = time.time()
        # Apply NMS
        pred = non_max_suppression(res_tensor, conf_thres=0.33, iou_thres=0.5, classes=None, agnostic=False)
        print("nms takes", time.time()-t)
        t = time.time()
        # Process detections
        bboxes = []
        src_img = img
        
        for i, det in enumerate(pred):  # detections per image
            # Rescale boxes from img_size to im0 size
            if det is not None:
                det[:, :4] = scale_coords((self.model_input_width, self.model_input_height), det[:, :4], src_img.shape).round()
                for *xyxy, conf, cls in det:
                    bboxes.append([*xyxy, conf, int(cls)])
            else:
                pass
        print("the rest takes", time.time()-t)
        return bboxes
    
    def run(self, img):
        
        img_resized = resize_image(img, (self.model_input_width, self.model_input_height))[:, :, ::-1]
        img_resized = (img_resized/255).astype(np.float32)
        img_resized = img_resized.transpose(2, 0, 1)
        
        img_resized = np.expand_dims(img_resized, axis=0)
        print("----", img_resized.shape)
        
        img_focused = focus_process(img_resized)
        print("----", img_focused.shape)
        
        img_dev_ptr, img_buf_size = self.transfer_img_to_device(img_focused)
#         print("img_dev_ptr, img_buf_size: ", img_dev_ptr, img_buf_size)
        self._gen_input_dataset(img_dev_ptr, img_buf_size)
        self.forward()
        
        ret = acl.rt.free(img_dev_ptr)
        check_ret("acl.rt.free", ret)
        
        
        pred_sbbox = get_model_output_by_index(self.output_data, 0)
        pred_mbbox = get_model_output_by_index(self.output_data, 1)
        pred_lbbox = get_model_output_by_index(self.output_data, 2)
        
        
        
        return [pred_sbbox, pred_mbbox, pred_lbbox]
#         return []
        self.pred_bbox = np.concatenate([pred_sbbox, \
                                    pred_mbbox, \
                                    pred_lbbox], axis=-1)
        
        print("pred_bbox shape", self.pred_bbox.shape)
        original_image_size = img.shape[:2]
        print("original_image_size", original_image_size)
#         bboxes = postprocess_boxes(self.pred_bbox, original_image_size, self.model_input_width, 0.3)
#         print(bboxes)
#         bboxes = nms(bboxes, 0.3, method='nms')
#         print(bboxes)
        return []
        return bboxes

    def forward(self):
        print('[Model] execute stage:')
        ret = acl.mdl.execute(self.model_id,
                              self.input_dataset,
                              self.output_data)
        check_ret("acl.mdl.execute", ret)

        #free the input dataset
        if self.input0_dataset_buffer:
            ret = acl.destroy_data_buffer(self.input0_dataset_buffer)
            check_ret("acl.destroy_data_buffer", ret)
            self.input0_dataset_buffer = None
        if self.input_dataset:
            ret = acl.mdl.destroy_dataset(self.input_dataset)
            check_ret("acl.destroy_dataset", ret)
            self.input_dataset = None

        print('[Model] execute stage success')

    def _gen_input_dataset(self, dvpp_output_buffer, dvpp_output_size, image_height=None, image_width=None):
        print("[Model] create model input dataset:")
        self.input_dataset = acl.mdl.create_dataset()
        self.input0_dataset_buffer = acl.create_data_buffer(dvpp_output_buffer,
                                                      dvpp_output_size)
        _, ret = acl.mdl.add_dataset_buffer(
            self.input_dataset,
            self.input0_dataset_buffer)
        if ret:
            ret = acl.destroy_data_buffer(self.input0_dataset_buffer)
            self.input0_dataset_buffer = None
            check_ret("acl.destroy_data_buffer", ret)

        print("[Model] create model input dataset success")

    def _gen_output_dataset(self, size):
        print("[Model] create model output dataset:")
        dataset = acl.mdl.create_dataset()
        for i in range(size):
            temp_buffer_size = acl.mdl.\
                get_output_size_by_index(self.model_desc, i)
            temp_buffer, ret = acl.rt.malloc(temp_buffer_size,
                                             ACL_MEM_MALLOC_NORMAL_ONLY)
            check_ret("acl.rt.malloc", ret)
            dataset_buffer = acl.create_data_buffer(
                temp_buffer,
                temp_buffer_size)

            _, ret = acl.mdl.add_dataset_buffer(dataset, dataset_buffer)
            if ret:
                acl.destroy_data_buffer(dataset)
                check_ret("acl.destroy_data_buffer", ret)
        self.output_data = dataset
        print("[Model] create model output dataset success")
        
    def _release_dataset(self, ):
        for dataset in [self.input_dataset, self.output_data]:
            if not dataset:
                continue
            num = acl.mdl.get_dataset_num_buffers(dataset)
            for i in range(num):
                data_buf = acl.mdl.get_dataset_buffer(dataset, i)
                if data_buf:
                    ret = acl.destroy_data_buffer(data_buf)
                    check_ret("acl.destroy_data_buffer", ret)

            ret = acl.mdl.destroy_dataset(dataset)
            check_ret("acl.mdl.destroy_dataset", ret)

