"""
Copyright (R) @huawei.com, all rights reserved
-*- coding:utf-8 -*-
CREATED:  2020-6-15 20:12:13
MODIFIED: 2020-6-15 14:04:45
"""
import numpy as np
import acl
from .constant import *
from .acl_util import check_ret
import cv2
import time

class Vdec():
    def __init__(self, device_id, input_width, input_height):
        self.device_id = device_id
        
        ret = acl.rt.set_device(self.device_id)
        check_ret("acl.rt.set_device", ret)
        # self.context, ret = acl.rt.create_context(self.device_id)
        # check_ret("acl.rt.create_context", ret)
        # self.stream, ret = acl.rt.create_stream()
        # check_ret("acl.rt.create_stream", ret)
        
        self.vdec_channel_desc = None
        self.input_width = input_width
        self.input_height = input_height
        self._vdec_exit = True
        self._en_type = H264_MAIN_LEVEL
        self._format = PIXEL_FORMAT_YVU_SEMIPLANAR_420
        self._channel_id = 0
        self.output_count = 0
        self.cb_thread_id = None
        self.images_buffer = []
        self.frame_config = acl.media.vdec_create_frame_config()
        
        self.decoded_img_buf_size = self.input_width * self.input_height * 3 // 2
            
    def __del__(self):
        print('[VDEC] release source stage:')


        # self._destroy_resource()
        # ret = acl.finalize()
#         check_ret("acl.finalize", ret)
        print('[VDEC] release source stage success')
        
    def _thread_func(self, args_list):
        self._vdec_exit = True
        timeout = args_list[0]
        
        thread_context, ret = acl.rt.create_context(self.device_id)
        check_ret("acl.rt.create_context", ret)
        while self._vdec_exit:
            acl.rt.process_report(timeout)
            
        ret = acl.rt.destroy_context(thread_context)
        check_ret("acl.rt.destroy_context", ret)
        
        print("[Vdec] [_thread_func] _thread_func out")

    def _callback(self, input_stream_desc, output_pic_desc, user_data):
        # input_stream_desc
        if input_stream_desc:
            ret = acl.media.dvpp_destroy_stream_desc(input_stream_desc)
            check_ret("acl.media.dvpp_destroy_stream_desc", ret)
            
        # output_pic_desc
        if output_pic_desc:
            
            vdec_out_buffer = acl.media.dvpp_get_pic_desc_data(output_pic_desc)
            ret = acl.media.dvpp_get_pic_desc_ret_code(output_pic_desc)
            check_ret("acl.media.dvpp_get_pic_desc_ret_code", ret)
            '''
            若单独执行视频解码操作，此处建议调用acl.rt.memcpy将解码后的数据拷贝回host侧
            在此样例中，因为之后还涉及到图片缩放与推理操作，所以此处不做数据拷贝
            '''
            # data_size = acl.media.dvpp_get_pic_desc_size(output_pic_desc)
            data_size = self.decoded_img_buf_size
            # print("data_size", data_size)
            # self.images_buffer.append(dict({"buffer": vdec_out_buffer,
            #                                 "size": data_size}))
            
            self.decoded_img_ptr_host, ret = acl.rt.malloc_host(self.decoded_img_buf_size)
            check_ret("acl.rt.malloc_host", ret)

            ret = acl.rt.memcpy(self.decoded_img_ptr_host, data_size, vdec_out_buffer, data_size, ACL_MEMCPY_DEVICE_TO_HOST)
            check_ret("acl.rt.memcpy", ret)
            
            decoded_img = acl.util.ptr_to_numpy(self.decoded_img_ptr_host, (data_size, ), 2)
            self.images_buffer.append(decoded_img)
            
            ret = acl.media.dvpp_free(vdec_out_buffer)
            check_ret("acl.media.dvpp_free", ret)
            
            # ret = acl.rt.free_host(self.decoded_img_ptr_host)
            # check_ret("acl.rt.free_host", ret)
            # decoded_img.tofile("img_%d"%self.output_count)
            
            ret = acl.media.dvpp_destroy_pic_desc(output_pic_desc)
            check_ret("acl.media.dvpp_destroy_pic_desc", ret)
            
            
        self.output_count += 1
        # print("[Vdec] [_callback] _callback exit success")

    def get_image_buffer(self):
        return self.images_buffer

    def init_resource(self):
        print("[Vdec] class Vdec init resource stage:")
        self.vdec_channel_desc = acl.media.vdec_create_channel_desc()
        
        # print("self.vdec_channel_desc", self.vdec_channel_desc)
        acl.media.vdec_set_channel_desc_channel_id(self.vdec_channel_desc,
                                                   self._channel_id)
        acl.media.vdec_set_channel_desc_thread_id(self.vdec_channel_desc,
                                                  self.cb_thread_id)
        acl.media.vdec_set_channel_desc_callback(self.vdec_channel_desc,
                                                 self._callback)
        acl.media.vdec_set_channel_desc_entype(self.vdec_channel_desc,
                                               self._en_type)
        acl.media.vdec_set_channel_desc_out_pic_format(self.vdec_channel_desc,
                                                       self._format)
        
        ret = acl.media.vdec_create_channel(self.vdec_channel_desc)
        check_ret("acl.media.vdec_create_channel", ret)
        
        print("[Vdec] class Vdec init resource stage success")
        
    def destroy_resource(self):
        print("[Vdec] release resource:")
        ret = acl.media.vdec_destroy_channel(self.vdec_channel_desc)
        check_ret("acl.media.vdec_destroy_channel", ret)
        ret = acl.media.vdec_destroy_channel_desc(self.vdec_channel_desc)
        check_ret("acl.media.vdec_destroy_channel_desc", ret)
        print("[Vdec] release resource success")
        
    def _set_pic_input(self, frame_dev_ptr, frame_size):
        self.dvpp_stream_desc = acl.media.dvpp_create_stream_desc()
        
        # print("self.dvpp_stream_desc", self.dvpp_stream_desc)
        ret = acl.media.dvpp_set_stream_desc_data(self.dvpp_stream_desc, frame_dev_ptr)
        check_ret("acl.media.dvpp_set_stream_desc_data", ret)
        ret = acl.media.dvpp_set_stream_desc_size(self.dvpp_stream_desc, frame_size)
        check_ret("acl.media.dvpp_set_stream_desc_size", ret)
        # print("[VDEC] create input stream desc success")

    def _set_pic_output(self):
        
        self.decoded_img_ptr_dev, ret = acl.media.dvpp_malloc(self.decoded_img_buf_size)
        check_ret("acl.media.dvpp_malloc", ret)
        
        self.dvpp_pic_desc = acl.media.dvpp_create_pic_desc()
        acl.media.dvpp_set_pic_desc_data(self.dvpp_pic_desc,
                                         self.decoded_img_ptr_dev)

        acl.media.dvpp_set_pic_desc_size(self.dvpp_pic_desc,
                                         self.decoded_img_buf_size)

        acl.media.dvpp_set_pic_desc_format(self.dvpp_pic_desc,
                                           self._format)
        # print("[Vdec] create output pic desc success")
        time.sleep(0.001)
        return

    def run(self, video_info):
        self.video_path, self.input_width, self.input_height, self.dtype = video_info
        self.images_buffer = []
        # 此处设置触发回调处理之前的等待时间，
        # 由acl.rt.subscribe_report接口指定的线程处理回调。
        timeout = 100
        self.cb_thread_id, ret = acl.util.start_thread(
            self._thread_func, [timeout])
        print("self.cb_thread_id", self.cb_thread_id)
        check_ret("acl.util.start_thread", ret)
        
        # ret = acl.rt.subscribe_report(self.cb_thread_id, self.stream)
        # check_ret("acl.rt.subscribe_report", ret)
        
        self.init_resource()
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_FORMAT, -1)
        self.read_frame_cnt = 0
        while cap.isOpened():
            
            ret, frame = cap.read()
            if ret:
                frame_size = frame.shape[1]
                frame_host_ptr = acl.util.numpy_to_ptr(frame)
                frame_dev_ptr, ret = acl.media.dvpp_malloc(frame_size)
                check_ret("acl.media.dvpp_malloc", ret)
                ret = acl.rt.memcpy(frame_dev_ptr,
                                    frame_size,
                                    frame_host_ptr,
                                    frame_size,
                                    ACL_MEMCPY_HOST_TO_DEVICE)
                check_ret("acl.rt.memcpy", ret)
                
                self._set_pic_input(frame_dev_ptr, frame_size)
                self._set_pic_output()
                
                # vdec_send_frame
                ret = acl.media.vdec_send_frame(self.vdec_channel_desc,
                                                self.dvpp_stream_desc,
                                                self.dvpp_pic_desc,
                                                self.frame_config,
                                                None)
                check_ret("acl.media.vdec_send_frame", ret)
                # print("sent %d frame(s)" % self.read_frame_cnt)
            else:
                break
        
        # self._vdec_exit = False
        # ret = acl.util.stop_thread(self.cb_thread_id)
        # check_ret("acl.util.stop_thread", ret)
        
        self.destroy_resource()
        print("[Vdec] vdec finish!!!\n")
