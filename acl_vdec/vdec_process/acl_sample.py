"""
Copyright (R) @huawei.com, all rights reserved
-*- coding:utf-8 -*-
CREATED:  2020-6-04 20:12:13
MODIFIED: 2020-9-30 09:00:00
"""
import os
import acl
import numpy as np
from acl_vdec import Vdec
from acl_util import check_ret
import cv2

class Sample():
    def __init__(self,
                 device_id):
        self.device_id = device_id  # int
        self.context = None  # pointer
        self.stream = None

        self.init_resource()
        self.vdec_process = Vdec(self.context,
                                 self.stream)

    def init_resource(self):
        print("init resource stage:")
#         acl.init()
        ret = acl.rt.set_device(self.device_id)
        check_ret("acl.rt.set_device", ret)

        self.context, ret = acl.rt.create_context(self.device_id)
        check_ret("acl.rt.create_context", ret)

        self.stream, ret = acl.rt.create_stream()
        check_ret("acl.rt.create_stream", ret)
        print("init resource stage success")

    def __del__(self):
        print('[Sample] release source stage:')

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
        # ret = acl.finalize()
        check_ret("acl.finalize", ret)
        print('[Sample] release source stage success')

    def forward(self, temp):
        # 视频解码过程
        self.vdec_process.run(temp)
        self.images_buffer = self.vdec_process.get_image_buffer()
        return
                
if __name__ == '__main__':
    MODEL_PATH = "./model/resnet50_aipp.om"
    VDEC_OUT_PATH = "./vdec_out"

    if not os.path.exists(VDEC_OUT_PATH):
        os.makedirs(VDEC_OUT_PATH)

    sample = Sample(0, MODEL_PATH, VDEC_OUT_PATH, 224, 224)
    vedio_list = ["./data/vdec_h265_1frame_rabbit_1280x720.h265",
                  1280,  # width
                  720,  # height
                  np.uint8]  # dtype
    sample.forward(vedio_list)
