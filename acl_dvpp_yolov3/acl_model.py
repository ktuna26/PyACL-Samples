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
    ACL_ERROR_NONE, NPY_BYTE
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
        self.input0_dataset_buffer = None
        self.input1_dataset_buffer = None
        self.input1_buffer = None

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

    def init_resource(self):
        print("[Model] class Model init resource stage:")
        # context
        acl.rt.set_context(self.context)
        # load_model
        self.model_id, ret = acl.mdl.load_from_file(self.model_path)
        check_ret("acl.mdl.load_from_file", ret)
        self.model_desc = acl.mdl.create_desc()
        print("结构",self.model_desc)
        ret = acl.mdl.get_desc(self.model_desc, self.model_id)
        check_ret("acl.mdl.get_desc", ret)
        
        input_size = acl.mdl.get_num_inputs(self.model_desc)
        print("input number:%d" % input_size)
        input1_size = acl.mdl.get_input_size_by_index(self.model_desc, 1)
        print("input %d: %d" % (1, input1_size))
        
        self.input1_buffer, ret = acl.rt.malloc(input1_size,
                                            ACL_MEM_MALLOC_NORMAL_ONLY)
        check_ret("acl.rt.malloc", ret)
        self.input1_dataset_buffer = acl.create_data_buffer(
            self.input1_buffer,
            input1_size)
        
        output_size = acl.mdl.get_num_outputs(self.model_desc)
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

    def run(self, dvpp_output_buffer, dvpp_output_size, image_height=None, image_width=None):
        self._gen_input_dataset(dvpp_output_buffer, dvpp_output_size, image_height, image_width)
        self.forward()
        self._print_result(self.output_data)
        return self.output_data

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

        if image_height != None and image_width != None:
            input2 = np.array([416, 416, image_height, image_width], dtype=np.float32)
            print("input2 {0}, size:{1}".format(input2, input2.size))
            input2_ptr = acl.util.numpy_to_ptr(input2)
            acl.rt.memcpy(self.input1_buffer, input2.size * input2.itemsize, input2_ptr,
                            input2.size * input2.itemsize, ACL_MEMCPY_HOST_TO_DEVICE)

            # Don't need to create the input1_dataset_buffer, because in init_resource(), create the self.input1_dataset_buffer with self.input1_buffer
            _, ret = acl.mdl.add_dataset_buffer(
                self.input_dataset,
                self.input1_dataset_buffer)
            check_ret("acl.add_dataset_buffer", ret)
            if ret:
                ret = acl.destroy_data_buffer(self.input1_dataset_buffer)
                self.input1_dataset_buffer = None    //判断是否清空了buffer
                check_ret("acl.destroy1_data_buffer", ret)
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
        dataset = {}
        res_num = 0
        num = acl.mdl.get_dataset_num_buffers(infer_output)
        for i in [1, 0]:
            temp_output_buf = acl.mdl.get_dataset_buffer(infer_output, i)

            infer_output_ptr = acl.get_data_buffer_addr(temp_output_buf)
            infer_output_size = acl.get_data_buffer_size(temp_output_buf)

            output_host, _ = acl.rt.malloc_host(infer_output_size)
            acl.rt.memcpy(output_host, infer_output_size, infer_output_ptr,
                          infer_output_size, ACL_MEMCPY_DEVICE_TO_HOST)

            # 对输出进行处理，将2种类别的输出分别分开
            if i == 1:
                result = acl.util.ptr_to_numpy(output_host,
                                           (infer_output_size//4,),# 因为要解析为int类型的数据，所以这里的size=byte_num/4
                                           6)# int32
                res_num = int(result[0])
                print("result ouput",res_num)
                dataset['num_detections'] = res_num
            elif i == 0:
                result = acl.util.ptr_to_numpy(output_host,
                                           (infer_output_size//4,),# 因为要解析为float32类型的数据，所以这里的size=byte_num/4
                                           11)# float32
                for j in range(res_num):
                    object = {}
                    object['x1'] = result[0 * res_num + j]
                    object['y1'] = result[1 * res_num + j]
                    object['x2'] = result[2 * res_num + j]
                    object['y2'] = result[3 * res_num + j]
                    object['detection_scores']  = float(result[4 * res_num + j])
                    object['detection_classes'] = result[5 * res_num + j]
                    print(object)
                    dataset[j] = object
                print("result",result)
           
            # free the host buffer
            ret= acl.rt.free_host(output_host)

        # 对推理结果进行打印
        print('[RESULT] ','num_detections: ', res_num)
        for i in range(res_num):
            print('[RESULT] ','result: ', i + 1)
            print('[RESULT] ','detection_classes: ', dataset[i]['detection_classes'])
            print('[RESULT] ','detection_scores: ', dataset[i]['detection_scores'])
            print('[RESULT] ','detection_boxes: ', dataset[i]['x1'], dataset[i]['y1'], dataset[i]['x2'], dataset[i]['y2'])
            print("dataset",dataset)