"""
Copyright 2021 Huawei Technologies Co., Ltd

CREATED:  2020-06-04 20:12:13
MODIFIED: 2021-12-27 27:48:45
"""

# -*- coding:utf-8 -*-
import acl
import cv2
import time
import numpy as np

from acl_util import check_ret
from constant import ACL_MEM_MALLOC_NORMAL_ONLY, \
    ACL_MEMCPY_HOST_TO_DEVICE, ACL_MEMCPY_DEVICE_TO_HOST, \
    ACL_ERROR_NONE, NPY_BYTE


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
        self.model_output_width = None
        self.model_output_height = None
        self.input_dataset = None
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
            self.model_input_height, self.model_input_width = acl.mdl.get_input_dims(self.model_desc, i)[0]['dims'][1:3]
        print("=" * 50)
        print("model output size", output_size)
        for i in range(output_size):
            print("output ", i)
            print("model output dims", acl.mdl.get_output_dims(self.model_desc, i))
            print("model output datatype", acl.mdl.get_output_data_type(self.model_desc, i))
            self.model_output_height, self.model_output_width = acl.mdl.get_output_dims(self.model_desc, i)[0]['dims']
        print("=" * 50)
        print("[Model] class Model init resource stage success")
        
        
    def transfer_img_to_device(self, img_resized):
        
        # BGR to RGB
        img_host_ptr = acl.util.numpy_to_ptr(img_resized)
        img_buf_size = img_resized.itemsize * img_resized.size
        print("[ACL] img_host_ptr, img_buf_size: ", img_host_ptr, img_buf_size)
        img_dev_ptr, ret = acl.rt.malloc(img_buf_size, ACL_MEM_MALLOC_NORMAL_ONLY)
        check_ret("acl.rt.malloc", ret)
        ret = acl.rt.memcpy(img_dev_ptr, img_buf_size, img_host_ptr, img_buf_size,
                                        ACL_MEMCPY_HOST_TO_DEVICE)
        check_ret("acl.rt.memcpy", ret)
        
        return img_dev_ptr, img_buf_size
    
    
    def resize_image(self, img, size):

        h, w = img.shape[:2]
        c = img.shape[2] if len(img.shape)>2 else 1

        if h == w: 
            return cv2.resize(img, size, cv2.INTER_AREA)

        dif = h if h > w else w

        interpolation = cv2.INTER_AREA if dif > (size[0]+size[1])//2 else \
                        cv2.INTER_CUBIC

        x_pos = (dif - w)//2
        y_pos = (dif - h)//2

        if len(img.shape) == 2:
            mask = np.zeros((dif, dif), dtype=img.dtype)
            mask.fill(128)
            mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
        else:
            mask = np.zeros((dif, dif, c), dtype=img.dtype)
            mask.fill(128)
            mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]

        return cv2.resize(mask, size, interpolation)
    
    
    def run(self, img):
        t0 = time.time()
        print("[INFO] classifying face . . .")
        
        img_resized =  self.resize_image(img,(self.model_input_width, 
                                                              self.model_input_height ))[:, :, ::-1]
        img_resized = img_resized.transpose(2, 0, 1)  # BGR to RGB, to 3x640x640
        
        image_np = np.array(img_resized, dtype=np.float32) / 255.0
        image_np_expanded = np.expand_dims(image_np, axis=0)  # NCHW

        print("[PreProc] img_resized shape", img_resized.shape)
        img_dev_ptr, img_buf_size = self.transfer_img_to_device(np.ascontiguousarray(image_np_expanded))
        print("[ACL] img_dev_ptr, img_buf_size: ", img_dev_ptr, img_buf_size)

        self._gen_input_dataset(img_dev_ptr, img_buf_size)
        self.forward()
        
        predict = self.get_model_output_by_index(0)
        ret = acl.rt.free(img_dev_ptr)
        check_ret("acl.rt.free", ret)
        
        print("[INFO] classification done!")
        t0 = time.time() - t0
        
        print("[Result] image runtime : {:.3f}".format(t0))

        return predict
        
        
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
        
        
    def _gen_input_dataset(self, dvpp_output_buffer, dvpp_output_size):
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
            
            
    def get_model_output_by_index(self, i):
        temp_output_buf = acl.mdl.get_dataset_buffer(self.output_data, i)

        infer_output_ptr = acl.get_data_buffer_addr(temp_output_buf)
        infer_output_size = acl.get_data_buffer_size(temp_output_buf)

        output_host, _ = acl.rt.malloc_host(infer_output_size)
        acl.rt.memcpy(output_host, infer_output_size, infer_output_ptr,
                              infer_output_size, ACL_MEMCPY_DEVICE_TO_HOST)
        
        return acl.util.ptr_to_numpy(output_host, (infer_output_size//4,), 11).reshape(-1, self.model_output_width)