"""
Copyright (R) @huawei.com, all rights reserved
-*- coding:utf-8 -*-
CREATED:  2020-6-04 20:12:13
MODIFIED: 2020-6-06 14:04:45
"""
import os
import acl
import numpy as np
from acl_vdec import Vdec
from acl_model import Model
from acl_util import check_ret
from acl_dvpp import Dvpp
from constant import ACL_MEMCPY_HOST_TO_DEVICE, ACL_ERROR_NONE


class Sample():
    def __init__(self,
                 device_id,
                 model_path,
                 vdec_out_path,
                 model_input_width,
                 model_input_height):
        self.device_id = device_id  # int
        self.model_path = model_path  # string
        self.context = None  # pointer
        self.stream = None
        self.model_input_width = model_input_width
        self.model_input_height = model_input_height,
        self.init_resource()
        self.model_process = Model(self.context,
                                   self.stream,
                                   model_path)
        self.vdec_process = Vdec(self.context,
                                 self.stream,
                                 vdec_out_path)
        self.dvpp_process = Dvpp(self.stream,
                                 model_input_width,
                                 model_input_height)
        self.model_input_width = model_input_width
        self.model_input_height = model_input_height
        self.vdec_out_path = vdec_out_path

    def init_resource(self):
        print("init resource stage:")
        acl.init()
        ret = acl.rt.set_device(self.device_id)
        check_ret("acl.rt.set_device", ret)

        self.context, ret = acl.rt.create_context(self.device_id)
        check_ret("acl.rt.create_context", ret)

        self.stream, ret = acl.rt.create_stream()
        check_ret("acl.rt.create_stream", ret)
        print("init resource stage success")

    def __del__(self):
        print('[Sample] release source stage:')
        if self.dvpp_process:
            del self.dvpp_process

        if self.model_process:
            del self.model_process

        if self.vdec_process:
            del self.vdec_process

        if self.stream:
            ret = acl.rt.destroy_stream(self.stream)
            check_ret("acl.rt.destroy_stream", ret)

        if self.context:
            ret = acl.rt.destroy_context(self.context)
            check_ret("acl.rt.destroy_context", ret)

        ret = acl.rt.reset_device(self.device_id)
        check_ret("acl.rt.reset_device", ret)
        ret = acl.finalize()
        check_ret("acl.finalize", ret)
        print('[Sample] release source stage success')

    def _transfer_to_devicce(self, img):
        img_ptr = img["buffer"]
        img_buffer_size = img["size"]
        img_device, ret = acl.media.dvpp_malloc(img_buffer_size)
        check_ret("acl.media.dvpp_malloc", ret)
        ret = acl.rt.memcpy(img_device,
                            img_buffer_size,
                            img_ptr,
                            img_buffer_size,
                            ACL_MEMCPY_HOST_TO_DEVICE)
        check_ret("acl.rt.memcpy", ret)

        ret = acl.rt.free_host(img_ptr)
        check_ret("acl.rt.free_host", ret)
        return img_device, img_buffer_size

    def forward(self, temp):
        _, input_width, input_height, _ = temp
        self.vdec_process.run(temp)

        images_buffer = self.vdec_process.get_image_buffer()
        if images_buffer:
            for img_buffer in images_buffer:
                img_device, img_buffer_size = \
                    self._transfer_to_devicce(img_buffer)
                dvpp_output_buffer, dvpp_output_size = \
                    self.dvpp_process.run(img_device,
                                          img_buffer_size,
                                          input_width,
                                          input_height)
                self.model_process.run(dvpp_output_buffer,
                                       dvpp_output_size)


if __name__ == '__main__':
    model_path = "./model/resnet50_aipp.om"
    vdec_out_path = "./vdec_out"

    if not os.path.exists(vdec_out_path):
        os.makedirs(vdec_out_path)
    device_id = 0
    sample = Sample(device_id, model_path, vdec_out_path, 224, 224)
    vedio_list = ["./data/vdec_h265_1frame_rabbit_1280x720.h265",
                   1280,  # width
                   720,     # height
                   np.uint8] #dtype

    sample.forward(vedio_list)
