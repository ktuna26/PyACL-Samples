"""
Copyright 2021 Huawei Technologies Co., Ltd

CREATED:  2020-06-04 20:12:13
MODIFIED: 2021-12-24 09:35:45
"""

# -*- coding:utf-8 -*-
import acl
import cv2
import numpy as np

from time import perf_counter
from acl_util import check_ret
from constant import ACL_MEM_MALLOC_NORMAL_ONLY, \
    ACL_MEMCPY_HOST_TO_DEVICE
from postprocessing import PostProcess


class Model(object):
    
    def __init__(self,
                 device_id,
                 model_path,
                 nms_thresh = 0.4):
        self.device_id = device_id
        self.model_path = model_path    # string
        self.nms_thresh = nms_thresh
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
        self.model_output_width = None
        self.model_output_height = None
        self.input_dataset = None
        self.model_output_dims = []
        self.model_input_element_number = None
        self.model_output_element_number = None
        self.postprocess = None
        self. __init_resource()
        
        
    def __del__(self):
        self.__release_dataset()
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
        
        
    def __init_resource(self):
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
        self. __gen_output_dataset(output_size)
        print("model input size", input_size)
        for i in range(input_size):
            print("input ", i)
            print("model input dims", acl.mdl.get_input_dims(self.model_desc, i))
            print("model input datatype", acl.mdl.get_input_data_type(self.model_desc, i))
            self.model_input_height, self.model_input_width = acl.mdl.get_input_dims(self.model_desc, i)[0]['dims'][2:]
            self.model_input_element_number = acl.mdl.get_input_dims(self.model_desc, i)[0]['dimCount']
        print("=" * 60)
        print("model output size", output_size)
        for i in range(output_size):
            print("output ", i)
            print("model output dims", acl.mdl.get_output_dims(self.model_desc, i))
            print("model output datatype", acl.mdl.get_output_data_type(self.model_desc, i))
            self.model_output_dims.append(acl.mdl.get_output_dims(self.model_desc, i)[0]['dims'])
            self.model_output_element_number = acl.mdl.get_output_dims(self.model_desc, i)[0]['dimCount']
        print("=" * 60)
        
        # set inital parameters according to model input size
        batched = False
        if self.model_output_element_number == 3:
            batched = True
            
        # set initial parameters according to model output size
        use_kps = False
        num_anchors = feat_stride_fpn = fmc = None
        if output_size == 6:
            fms = 1
            fmb = 2
            feat_stride_fpn = [8, 16, 32]
            num_anchors = 2
        elif output_size == 9:
            fms = 1
            fmb = 2
            feat_stride_fpn = [8, 16, 32]
            num_anchors = 2
            use_kps = True
        elif output_size == 10:
            fms = 1
            fmb = 2
            feat_stride_fpn = [8, 16, 32, 64, 128]
            num_anchors = 1
        elif output_size == 15:
            fms = 1
            fmb = 2
            feat_stride_fpn = [8, 16, 32, 64, 128]
            num_anchors = 1
            use_kps = True
        
        # creat postprocess object 
        self.postprocess = PostProcess (self.model_output_dims, feat_stride_fpn, self.nms_thresh, 
                                                            fms, fmb, num_anchors, batched, use_kps)

        print("[Model] class Model init resource stage success")
        
        
    def __transfer_img_to_device(self, img_resized):
        
        # BGR to RGB
        img_host_ptr = acl.util.numpy_to_ptr(img_resized)
        img_buf_size = img_resized.itemsize * img_resized.size
        print("[ACL] img_host_ptr, img_buf_size: ", img_host_ptr, img_buf_size)
        img_dev_ptr, ret = acl.rt.malloc(img_buf_size, ACL_MEM_MALLOC_NORMAL_ONLY)
        check_ret("acl.rt.malloc", ret)
        ret = acl.rt.memcpy(img_dev_ptr, img_buf_size, img_host_ptr, img_buf_size, ACL_MEMCPY_HOST_TO_DEVICE)
        check_ret("acl.rt.memcpy", ret)
        
        return img_dev_ptr, img_buf_size
    
    
    def run(self, img, thresh = 0.5):
        im_ratio = float(img.shape[0]) / img.shape[1]
        
        input_size = self.model_input_width, self.model_input_height
        model_ratio = float(input_size[1]) / input_size[0]
        
        if im_ratio>model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        
        t0 = perf_counter()
        # resize   
        det_scale = float(new_height) / img.shape[0]
        resized_img = cv2.resize(img, (new_width, new_height))
        det_img = np.zeros( (input_size[1], input_size[0], 3), dtype=np.uint8 )
        det_img[:new_height, :new_width, :] = resized_img
        
        input_size = tuple(det_img.shape[0:2][::-1])
        blob = cv2.dnn.blobFromImage(det_img, 1.0/128, input_size, (127.5, 127.5, 127.5), swapRB=True)
        input_height = blob.shape[2]
        input_width = blob.shape[3]

        # preprocessing
        print("[PreProc] image_np_expanded shape:", blob.shape)
        img_resized = np.ascontiguousarray(blob)
        
        img_dev_ptr, img_buf_size = self.__transfer_img_to_device(img_resized)
        print("[ACL] img_dev_ptr, img_buf_size: ", img_dev_ptr, img_buf_size)
        
        self. __gen_input_dataset(img_dev_ptr, img_buf_size)
        self. __forward()
        
        ret = acl.rt.free(img_dev_ptr)
        check_ret("acl.rt.free", ret)
        
        # postprocessing
        bboxes, kpss = self.postprocess.detect(self.output_data, det_scale, input_height, input_width, thresh)
        t1 = perf_counter()
        
        print("[Result] infer time : %.3f ms" % ((t1 - t0) * 1000))
        
        return bboxes, kpss
        
        
    def __forward(self):
        print('[Model] execute stage:')
        ret = acl.mdl.execute(self.model_id,
                              self.input_dataset,
                              self.output_data)
        #check_ret("acl.mdl.execute", ret)

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
        
        
    def __gen_input_dataset(self, dvpp_output_buffer, dvpp_output_size, image_height=None, image_width=None):
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
        
        
    def __gen_output_dataset(self, size):
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
        
        
    def __release_dataset(self, ):
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