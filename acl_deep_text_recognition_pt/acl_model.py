"""
Copyright 2021 Huawei Technologies Co., Ltd

CREATED:  2020-6-04 20:12:13
MODIFIED: 2021-11-19 00:48:45
"""

# -*- coding:utf-8 -*-
import acl
import time
import cv2
import numpy as np
from functools import cmp_to_key

from acl_util import check_ret
from postprocessing import Process
from box_utils import BBox, compare_alldims, compare_titledims
from constant import ACL_MEM_MALLOC_NORMAL_ONLY, \
    ACL_MEMCPY_HOST_TO_DEVICE, ACL_MEMCPY_DEVICE_TO_HOST


class Model(object):
    
    def __init__(self,
                 device_id,
                 model_path,
                character):
        self.device_id = device_id
        self.model_path = model_path    # string
        self.char_list =  ['_'] + list(character)    #decoded characters
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
            self.model_input_height, self.model_input_width = acl.mdl.get_input_dims(self.model_desc, i)[0]['dims'][2:]
        print("=" * 50)
        print("model output size", output_size)
        for i in range(output_size):
            print("output ", i)
            print("model output dims", acl.mdl.get_output_dims(self.model_desc, i))
            print("model output datatype", acl.mdl.get_output_data_type(self.model_desc, i))
            self.model_output_height, self.model_output_width = acl.mdl.get_output_dims(self.model_desc, i)[0]['dims'][1:3]
        print("=" * 50)
        print("[Model] class Model init resource stage success")
        
        
    def run(self, path_dict):
        # load image & boxes
        org_img = cv2.imread(path_dict['img_path'])
        bboxes_sorted = self._read_bboxes(path_dict['boxes_path'], org_img)

        # pre-processing
        t0 = time.time()
        print("[INFO] recognizing texts . . .")
              
        # crop textboxes from image
        cropped_imgs = []
        for box in bboxes_sorted:
            # crop each rectangle, enlarged by horizontal offset value, from original image
            cropped_imgs.append(box.get_crop(0, 0))
        
        process = Process(self.char_list, bboxes_sorted)
        for i, cropped_img in enumerate(cropped_imgs):
            img_gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)/127.5 - 1.0
            img_resized = cv2.resize(img_gray,(self.model_input_width, self.model_input_height ),interpolation=cv2.INTER_CUBIC)
            img_np_expanded = np.expand_dims(np.float32(img_resized), (0,1))

            print("[PreProc] image_np_expanded shape:", img_np_expanded.shape)
            img_dev_ptr, img_buf_size = self.transfer_img_to_device( np.ascontiguousarray(
                                                                                                            img_np_expanded))
            print("[ACL] img_dev_ptr, img_buf_size: ", img_dev_ptr, img_buf_size)

            self._gen_input_dataset(img_dev_ptr, img_buf_size)
            self.forward()

            ret = acl.rt.free(img_dev_ptr)
            check_ret("acl.rt.free", ret)

            preds = self.get_model_output_by_index(0)
            
        # Post-processing
            process.text_boxes(preds, i)    # recognize corresponding text for each text box

        print("[INFO] recognition done . . .")
            
        if len(bboxes_sorted)==0:
            return len(bboxes_sorted)

        # concatenate text boxes that mistakenly detected as two words but actually is one word
        bboxes_sorted = process.get_concat_same() 
        
        t0 = time.time() - t0
        print("[Result] image runtime : {:.3f}".format(t0))

        return bboxes_sorted
    
    
    def _read_bboxes(self, boxes_path, image):
        # read coordinates of extracted bounding boxes from text file
        with open(boxes_path, 'r') as bbox_f:
            box_coords = [line.rstrip('\n') for line in bbox_f]
        vertical_positions = np.argsort(np.asarray([int(y0.split(',')[1]) for y0 in box_coords]))
        
        # create bounding box objects
        bboxes = []
        
        # save box coordinates from EAST output which is pixel location of each vertex of rectangle
        print("[INFO] reading text boxes . . .")
        for i in vertical_positions:
            line = box_coords[i]
            box_coord = [int(i) for i in line.split(',')]
            bboxes.append(BBox(box_coord, 1, 0, image))
        bboxes_sorted = sorted(bboxes, key=cmp_to_key(compare_alldims))
        
        print('[INFO]  {} text boxes found.'.format(len(bboxes_sorted)))
        
        return bboxes_sorted
    
    
    def transfer_img_to_device(self, img_resized):
        
        # BGR to RGB
        img_host_ptr = acl.util.numpy_to_ptr(img_resized)
        img_buf_size = img_resized.itemsize * img_resized.size
        print("[ACL] img_host_ptr, img_buf_size: ", img_host_ptr, img_buf_size)
        img_dev_ptr, ret = acl.rt.malloc(img_buf_size, ACL_MEM_MALLOC_NORMAL_ONLY)
        check_ret("acl.rt.malloc", ret)
        ret = acl.rt.memcpy(img_dev_ptr, img_buf_size, img_host_ptr, img_buf_size, ACL_MEMCPY_HOST_TO_DEVICE)
        check_ret("acl.rt.memcpy", ret)
        
        return img_dev_ptr, img_buf_size
    
    
    def forward(self):
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
            
            
    def get_model_output_by_index(self, i):
        temp_output_buf = acl.mdl.get_dataset_buffer(self.output_data, i)

        infer_output_ptr = acl.get_data_buffer_addr(temp_output_buf)
        infer_output_size = acl.get_data_buffer_size(temp_output_buf)

        output_host, _ = acl.rt.malloc_host(infer_output_size)
        acl.rt.memcpy(output_host, infer_output_size, infer_output_ptr,
                              infer_output_size, ACL_MEMCPY_DEVICE_TO_HOST)
        
        return acl.util.ptr_to_numpy(output_host, (infer_output_size//4,), 11).reshape(-1,  self.model_output_height, 
                                                                                       self.model_output_width)