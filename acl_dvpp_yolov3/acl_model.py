"""
Copyright (R) @huawei.com, all rights reserved
-*- coding:utf-8 -*-
CREATED:  2020-6-04 20:12:13
MODIFIED: 2020-6-28 14:04:45
"""
import acl
import struct
import numpy as np
from constant import ACL_MEM_MALLOC_NORMAL_ONLY, \
    ACL_MEMCPY_HOST_TO_DEVICE, ACL_MEMCPY_DEVICE_TO_HOST, \
    ACL_ERROR_NONE, NPY_BYTE, ACL_MEMCPY_DEVICE_TO_DEVICE
from acl_util import check_ret


class Model(object):
    def __init__(self,
                 context,
                 stream,
                 model_path,
                 ):
        self.model_path = model_path    # string
        self.model_id = None            # pointer
        self.context = None             # pointer
        self.context = context  # pointer
        self.stream = stream
        self.input_data = None
        self.output_data = None
        self.model_desc = None          # pointer when using
        self.input_dataset = None
        self.output_dataset = None
        self.init_resource()

    def __del__(self):
        self._release_dataset()
        if self.model_id:
            ret = acl.mdl.unload(self.model_id)
            check_ret("acl.mdl.unload", ret)

        if self.model_desc:
            ret = acl.mdl.destroy_desc(self.model_desc)
            check_ret("acl.mdl.destroy_desc", ret)

        print("[Model] class Model releases resources successfully")

    def init_resource(self):
        print("[Model] class Model init resource stage:")
        # context
        acl.rt.set_context(self.context)
        # load_model
        self.model_id, ret = acl.mdl.load_from_file(self.model_path)
        check_ret("acl.mdl.load_from_file", ret)
        self.model_desc = acl.mdl.create_desc()
        ret = acl.mdl.get_desc(self.model_desc, self.model_id)
        check_ret("acl.mdl.get_desc", ret)
        output_size = acl.mdl.get_num_outputs(self.model_desc)
        print("model output size", output_size)
        for i in range(output_size):
            print("output ", i)
            print("model output dims", acl.mdl.get_output_dims(self.model_desc, i))
            print("model output datatype", acl.mdl.get_output_data_type(self.model_desc, i))
        
        input_size = acl.mdl.get_num_inputs(self.model_desc)
        print("=" * 25)
        print("model input size", input_size)
        for i in range(input_size):
            print("input ", i)
            print("model input dims", acl.mdl.get_input_dims(self.model_desc, i))
            print("model input datatype", acl.mdl.get_input_data_type(self.model_desc, i))
        
        self._gen_output_dataset(output_size)
        print("[Model] class Model init resource stage success")

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

    def run(self, dvpp_output_buffer, dvpp_output_size):
        self._gen_input_dataset(dvpp_output_buffer, dvpp_output_size)
        self.forward()
        #self._print_result(self.output_data)
        return self.output_data

    def forward(self):
        print('[Model] execute stage:')
        ret = acl.mdl.execute(self.model_id,
                              self.input_dataset,
                              self.output_data)
        check_ret("acl.mdl.execute", ret)
        print('[Model] execute stage success')

    def _gen_input_dataset(self, dvpp_output_buffer, dvpp_output_size):
        print("[Model] create model input dataset:")
        self.input_dataset = acl.mdl.create_dataset()
        input_dataset_buffer = acl.create_data_buffer(dvpp_output_buffer,
                                                      dvpp_output_size)
        _, ret = acl.mdl.add_dataset_buffer(
            self.input_dataset,
            input_dataset_buffer)
        if ret:
            ret = acl.destroy_data_buffer(self.input_dataset)
            check_ret("acl.destroy_data_buffer", ret)
           
        img_info, ret = acl.rt.malloc(4 * 4, ACL_MEM_MALLOC_NORMAL_ONLY)
        check_ret("acl.rt.malloc", ret)
        ret = acl.rt.memcpy(img_info, 16, \
                      acl.util.numpy_to_ptr(np.array([416, 416, 576, 768], dtype=np.float32)), \
                      16, \
                      ACL_MEMCPY_HOST_TO_DEVICE)
        check_ret("acl.rt.memcpy", ret)
        imginfo_dataset_buffer = acl.create_data_buffer(img_info, 16)
        _, ret = acl.mdl.add_dataset_buffer(self.input_dataset, imginfo_dataset_buffer)
        check_ret("acl.mdl.add_dataset_buffer", ret)
        
        print("[Model] create model input dataset success")

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

    def _print_result(self, infer_output):
        num = acl.mdl.get_dataset_num_buffers(infer_output)
        for i in range(num):
            temp_output_buf = acl.mdl.get_dataset_buffer(infer_output, i)

            infer_output_ptr = acl.get_data_buffer_addr(temp_output_buf)
            infer_output_size = acl.get_data_buffer_size(temp_output_buf)
            print("infer_output_size", infer_output_size)
            output_host, _ = acl.rt.malloc_host(infer_output_size)
            acl.rt.memcpy(output_host, infer_output_size, infer_output_ptr,
                          infer_output_size, ACL_MEMCPY_DEVICE_TO_HOST)
            result = acl.util.ptr_to_numpy(output_host,
                                           (infer_output_size,),
                                           NPY_BYTE)
            st = struct.unpack("1000f", bytearray(result))
            vals = np.array(st).flatten()
            top_k = vals.argsort()[-1:-6:-1]
            possible = 0
            print("\n======== top5 inference results: =============")
            for n in top_k:
                print("label:%d  prob: %f" % (n, vals[n]))
                possible += vals[n]
            # print("result: class_type:{}, top1:{:f}, top5:{:f} "
            #       .format(top_k[0], vals[top_k[0]], possible))
            ret = acl.rt.free_host(output_host)
            check_ret("acl.rt.free_host", ret)
