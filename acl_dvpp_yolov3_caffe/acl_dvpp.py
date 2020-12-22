"""
Copyright (R) @huawei.com, all rights reserved
-*- coding:utf-8 -*-
CREATED:  2020-6-04 20:12:13
MODIFIED: 2020-6-28 14:04:45
"""
import acl
from constant import PIXEL_FORMAT_YUV_SEMIPLANAR_420
from acl_util import check_ret
import numpy as np

class Dvpp():
    def __init__(self, stream, model_input_width, model_input_height):
        self._dvpp_channel_desc = None
        self._resize_config = None
        self._decode_out_dev_buffer = None
        self._decode_output_desc_ = None
        self._resize_input_desc_ = None
        self._resize_out_desc = None
        self._in_dev_buffer_ = None
        self._in_dev_buffer_size = 0
        self._input_width = 0
        self._input_height = 0
        self._resize_out_dev = None
        self._resize_out_size = 0
        self.stream = stream
        self._format = PIXEL_FORMAT_YUV_SEMIPLANAR_420
        self._model_input_width = model_input_width
        self._model_input_height = model_input_height

        self._cropArea = None
        self._pasteArea = None

        self.init_resource()

    def __del__(self):
        if self._decode_out_dev_buffer:
            acl.media.dvpp_free(self._decode_out_dev_buffer)
            self._decode_out_dev_buffer = None

        if self._resize_input_desc_:
            acl.media.dvpp_destroy_pic_desc(self._resize_input_desc_)
            self._resize_input_desc_ = None

        if self._resize_out_desc:
            acl.media.dvpp_destroy_pic_desc(self._resize_out_desc)
            self._resize_out_desc = None

        if self._resize_config:
            acl.media.dvpp_destroy_resize_config(self._resize_config)

        if self._dvpp_channel_desc:
            acl.media.dvpp_destroy_channel(self._dvpp_channel_desc)
            acl.media.dvpp_destroy_channel_desc(self._dvpp_channel_desc)

        if self._resize_out_dev:
            acl.media.dvpp_free(self._resize_out_dev)
            self._resize_out_dev = None

        if self._cropArea:
            acl.media.dvpp_destroy_roi_config(self._cropArea)
    
        if self._pasteArea:
            acl.media.dvpp_destroy_roi_config(self._pasteArea)
        print("[Dvpp] class Dvpp exist success")

    def init_resource(self):
        self._dvpp_channel_desc = acl.media.dvpp_create_channel_desc()
        acl.media.dvpp_create_channel(self._dvpp_channel_desc)
        self._resize_config = acl.media.dvpp_create_resize_config()

        self._cropArea = acl.media.dvpp_create_roi_config(0, 1, 0, 1)
        self._pasteArea = acl.media.dvpp_create_roi_config(0, 1, 0, 1)
    
    def release_buffer(self):
        if self._decode_out_dev_buffer:
            acl.media.dvpp_free(self._decode_out_dev_buffer)
            self._decode_out_dev_buffer = None

        if self._resize_out_dev:
            acl.media.dvpp_free(self._resize_out_dev)
            self._resize_out_dev = None

    def get_output(self):
        return self._resize_out_dev, self._resize_out_size

    def gen_tensor_desc(self,
                        temp_buffer,
                        temp_width,
                        temp_height,
                        flag=True,
                        need_malloc=True):
        if flag:
            decode_out_width = int(int((temp_width + 127) / 128) * 128)
            decode_out_height = int(int((temp_height + 15) / 16) * 16)
        else:
            decode_out_width = int(int((temp_width + 15) / 16) * 16)
            decode_out_height = int(int((temp_height + 1) / 2) * 2)
        # JPEGD的输入内存和输出内存要求128对齐，详见手册
        # VPC输入输出要求宽16对齐，高2对齐 
        # YUV420SP  buffer_size = width*height*3/2
        decode_out_buffer_size = decode_out_width*decode_out_height*3//2
        print("decode_out_buffer_size:",decode_out_buffer_size)
        print("temp_width:%d,temp_height:$d",temp_width,temp_height)
        if need_malloc:
            temp_buffer, ret = acl.media.dvpp_malloc(decode_out_buffer_size)
            check_ret("acl.media.dvpp_malloc", ret)
            acl.media.dvpp_free(decode_out_buffer_size)

        temp_desc = acl.media.dvpp_create_pic_desc()
        acl.media.dvpp_set_pic_desc_data(temp_desc, temp_buffer)
        acl.media.dvpp_set_pic_desc_format(temp_desc, self._format)
        acl.media.dvpp_set_pic_desc_width(temp_desc, temp_width)
        acl.media.dvpp_set_pic_desc_height(temp_desc, temp_height)
        acl.media.dvpp_set_pic_desc_width_stride(temp_desc, decode_out_width)
        acl.media.dvpp_set_pic_desc_height_stride(temp_desc, decode_out_height)
        acl.media.dvpp_set_pic_desc_size(temp_desc, decode_out_buffer_size)
        return temp_desc, temp_buffer, decode_out_buffer_size

    def gen_CropAndPaste_config(self, 
                                img_width,
                                img_height,
                                resize_width,
                                resize_height):
        resize_ratio = min(resize_width / img_width, resize_height / img_height)

        resize_w = int(resize_ratio * img_width)
        resize_h = int(resize_ratio * img_height)

        dw = int((resize_width - resize_w) / 2)
        dh = int((resize_height - resize_h) / 2)

        #ret = acl.media.dvpp_set_roi_config(self._cropArea, 
        #                                        0, 
        #                                        img_width if (img_width % 2) == 1 else (img_width - 1),
        #                                        0, 
        #                                        img_height if (img_height % 2) == 1 else (img_height -1))
        #check_ret("acl.media.dvpp_create_roi_config", ret)

        ret = acl.media.dvpp_set_roi_config_left(self._cropArea, 0)
        check_ret("acl.media.dvpp_create_roi_config_left", ret)

        ret = acl.media.dvpp_set_roi_config_right(self._cropArea, img_width if (img_width % 2) == 1 else (img_width - 1))
        check_ret("acl.media.dvpp_create_roi_config_right", ret)

        ret = acl.media.dvpp_set_roi_config_top(self._cropArea, 0)
        check_ret("acl.media.dvpp_create_roi_config_top", ret)

        ret = acl.media.dvpp_set_roi_config_bottom(self._cropArea, img_height if (img_height % 2) == 1 else (img_height -1))
        check_ret("acl.media.dvpp_create_roi_config_bottom", ret)

        #ret = acl.media.dvpp_set_roi_config(self._pasteArea, 
        #                                        dw if dw % 2 == 0 else (dw + 1),
        #                                        (dw + resize_w) if (dw + resize_w) % 2 == 1 else (dw + resize_w - 1),
        #                                        dh if dh % 2 == 0 else (dh + 1), 
        #                                        (dh + resize_h) if (dh + resize_h) % 2 == 1 else (dh + resize_h -1))
        #check_ret("acl.media.dvpp_create_roi_config", ret)

        ret = acl.media.dvpp_set_roi_config_left(self._pasteArea, dw if dw % 2 == 0 else (dw + 1))
        check_ret("acl.media.dvpp_create_roi_config_left", ret)

        ret = acl.media.dvpp_set_roi_config_right(self._pasteArea, (dw + resize_w) if (dw + resize_w) % 2 == 1 else (dw + resize_w - 1))
        check_ret("acl.media.dvpp_create_roi_config_right", ret)

        ret = acl.media.dvpp_set_roi_config_top(self._pasteArea, dh if dh % 2 == 0 else (dh + 1))
        check_ret("acl.media.dvpp_create_roi_config_top", ret)

        ret = acl.media.dvpp_set_roi_config_bottom(self._pasteArea, (dh + resize_h) if (dh + resize_h) % 2 == 1 else (dh + resize_h - 1))
        check_ret("acl.media.dvpp_create_roi_config_bottom", ret)
        print('[Dvpp] crop and paste roi config success')

    def _decode_process(self,
                        img_buffer,
                        img_buffer_size,
                        img_width,
                        img_height):
        print('[Dvpp] vpc decode stage:')
        self._decode_output_desc_, self._decode_out_dev_buffer, temp_size = \
            self.gen_tensor_desc(self._decode_out_dev_buffer,
                                 img_width, img_height)
        ret = acl.media.dvpp_jpeg_decode_async(self._dvpp_channel_desc,
                                               img_buffer,
                                               img_buffer_size,
                                               self._decode_output_desc_,
                                               self.stream)
        if ret != 0:
            raise Exception("dvpp_jpeg_decode_async failed ret={}".format(ret))
        acl.rt.synchronize_stream(self.stream)
        check_ret("acl.rt.synchronize_stream", ret)
        print('[Dvpp] vpc decode stage success')

    def _resize_process(self, img_width, img_height, type=None):
        print('[Dvpp] vpc resize stage:')
        self._resize_in_desc_, self._decode_out_buffer, _resize_in_size = \
            self.gen_tensor_desc(self._decode_out_dev_buffer,
                                 img_width,
                                 img_height,
                                 need_malloc=False)
        self._resize_out_desc, self._resize_out_dev, self._resize_out_size = \
            self.gen_tensor_desc(self._resize_out_dev,
                                 self._model_input_width,
                                 self._model_input_height,
                                 flag=False)
        
        if type == None:
            ret = acl.media.dvpp_vpc_resize_async(self._dvpp_channel_desc,
                                        self._resize_in_desc_,
                                        self._resize_out_desc,
                                        self._resize_config,
                                        self.stream)
            check_ret("acl.media.dvpp_vpc_resize_async", ret)
        else: #Proportional_Scale
            ret = acl.rt.memset(self._resize_out_dev, self._resize_out_size, 128, self._resize_out_size)
            check_ret("acl.media.memset", ret)
            self.gen_CropAndPaste_config(img_width, img_height, self._model_input_width, self._model_input_height)
            ret = acl.media.dvpp_vpc_crop_and_paste_async(self._dvpp_channel_desc,
                                        self._resize_in_desc_,
                                        self._resize_out_desc,
                                        self._cropArea,
                                        self._pasteArea,
                                        self.stream)
            check_ret("acl.media.dvpp_vpc_crop_and_paste_async", ret)

        ret = acl.rt.synchronize_stream(self.stream)
        check_ret("acl.rt.synchronize_stream", ret)
        print('[Dvpp] vpc resize stage success')    
        print(self._resize_out_dev, self._resize_out_size)

    def run(self,
            img_buffer,
            img_buffer_size,
            img_width,
            img_height):
        self._decode_process(img_buffer,
                             img_buffer_size,
                             img_width,
                             img_height)
        # free the decode output Description
        if self._decode_output_desc_:
            acl.media.dvpp_destroy_pic_desc(self._decode_output_desc_)
            self._decode_output_desc_ = None

        self._resize_process(img_width, img_height, "Proportional_Scale")
        # free the resize input and output Description
        if self._resize_input_desc_:
            acl.media.dvpp_destroy_pic_desc(self._resize_input_desc_)
            self._resize_input_desc_ = None
        if self._resize_out_desc:
            acl.media.dvpp_destroy_pic_desc(self._resize_out_desc)
            self._resize_out_desc = None

        return self._resize_out_dev, self._resize_out_size
