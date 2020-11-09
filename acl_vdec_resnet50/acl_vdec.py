"""
Copyright (R) @huawei.com, all rights reserved
-*- coding:utf-8 -*-
CREATED:  2020-6-15 20:12:13
MODIFIED: 2020-6-15 14:04:45
"""
import numpy as np
import os
import acl
from constant import ACL_MEM_MALLOC_NORMAL_ONLY, \
    ACL_MEMCPY_DEVICE_TO_HOST, ACL_MEMCPY_HOST_TO_DEVICE, \
    NPY_UBYTE, H265_MAIN_LEVEL, PIXEL_FORMAT_YVU_SEMIPLANAR_420
from acl_util import check_ret


class Vdec():
    def __init__(self, context, stream, vdec_out_path):
        self.context = context
        self.stream = stream
        self.vdec_out_path = vdec_out_path
        self.vdec_channel_desc = None
        self.stream_input_desc = None
        self.pic_output_desc = None
        self.pic_out_buffer = None
        self.input_width = 0
        self.input_height = 0
        self._vdec_exit = True
        self._en_type = H265_MAIN_LEVEL
        self._format = PIXEL_FORMAT_YVU_SEMIPLANAR_420
        self._channel_id = 10
        self._rest_len = 10
        self.output_count = 1
        self.rest_len = 10
        self.images_buffer = []

    def _thread_func(self, args_list):
        timeout = args_list[0]
        acl.rt.set_context(self.context)
        while self._vdec_exit:
            ret = acl.rt.process_report(timeout)
            check_ret("acl.rt.process_report", ret)
        print("[Vdec] [_thread_func] _thread_func out")

    def _callback(self, input_stream_desc, output_pic_desc, user_data):
        # input_stream_desc
        if input_stream_desc:
            ret = acl.media.dvpp_destroy_stream_desc(input_stream_desc)
            check_ret("acl.media.dvpp_destroy_stream_desc", ret)
        # output_pic_desc
        if output_pic_desc:
            vdec_out_buffer = acl.media.dvpp_get_pic_desc_data(output_pic_desc)
            ret_code = acl.media.dvpp_get_pic_desc_ret_code(output_pic_desc)

            if ret_code == 0:
                data_size = acl.media.dvpp_get_pic_desc_size(output_pic_desc)
                host_buffer, ret = acl.rt.malloc_host(data_size)
                check_ret("acl.rt.malloc_host", ret)

                ret = acl.rt.memcpy(host_buffer,
                                    data_size,
                                    vdec_out_buffer,
                                    data_size,
                                    ACL_MEMCPY_DEVICE_TO_HOST)
                check_ret("acl.rt.memcpy", ret)
                output_pic_numpy = acl.util.ptr_to_numpy(
                    host_buffer, (data_size,),
                    NPY_UBYTE)
                file_name = os.path.join(self.vdec_out_path,
                                         "images_{}".format(self.output_count))
                print("[Vdec]", file_name)
                np.save(file_name, output_pic_numpy)
                # acl.rt.free_host(host_buffer)
                self.images_buffer.append(dict({"buffer": host_buffer,
                                                "size": data_size}))
            ret = acl.media.dvpp_free(vdec_out_buffer)
            check_ret("acl.media.dvpp_free", ret)
            ret = acl.media.dvpp_destroy_pic_desc(output_pic_desc)
            check_ret("acl.media.dvpp_destroy_pic_desc", ret)
        self.output_count += 1
        print("[Vdec] [_callback] _callback exist success")

    def get_image_buffer(self):
        return self.images_buffer

    def init_resource(self, cb_thread_id):
        print("[Vdec] class Vdec init resource stage:")
        self.vdec_channel_desc = acl.media.vdec_create_channel_desc()
        acl.media.vdec_set_channel_desc_channel_id(self.vdec_channel_desc,
                                                   self._channel_id)
        acl.media.vdec_set_channel_desc_thread_id(self.vdec_channel_desc,
                                                  cb_thread_id)
        acl.media.vdec_set_channel_desc_callback(self.vdec_channel_desc,
                                                 self._callback)
        acl.media.vdec_set_channel_desc_entype(self.vdec_channel_desc,
                                               self._en_type)
        acl.media.vdec_set_channel_desc_out_pic_format(self.vdec_channel_desc,
                                                       self._format)
        acl.media.vdec_create_channel(self.vdec_channel_desc)
        print("[Vdec] class Vdec init resource stage success")

    def _gen_input_dataset(self, img_path):
        img = np.fromfile(img_path, dtype=self.dtype)
        img_buffer_size = img.size
        img_ptr = acl.util.numpy_to_ptr(img)
        img_device, ret = acl.media.dvpp_malloc(img_buffer_size)
        ret = acl.rt.memcpy(img_device,
                            img_buffer_size,
                            img_ptr,
                            img_buffer_size,
                            ACL_MEMCPY_HOST_TO_DEVICE)
        check_ret("acl.rt.memcpy", ret)
        return img_device, img_buffer_size

    def _set_input(self, input_stream_size):
        # print("create vdec input stream desc:")
        self.dvpp_stream_desc = acl.media.dvpp_create_stream_desc()
        ret = acl.media.dvpp_set_stream_desc_data(self.dvpp_stream_desc,
                                                  self.input_stream_mem)
        check_ret("acl.media.dvpp_set_stream_desc_data", ret)
        ret = acl.media.dvpp_set_stream_desc_size(self.dvpp_stream_desc,
                                                  input_stream_size)
        check_ret("acl.media.dvpp_set_stream_desc_size", ret)
        print("[Vdec] create input stream desc success")

    def _set_pic_output(self, output_pic_size):
        # pic_desc
        output_pic_mem, ret = acl.media.dvpp_malloc(output_pic_size)
        check_ret("acl.media.dvpp_malloc", ret)

        self.dvpp_pic_desc = acl.media.dvpp_create_pic_desc()
        acl.media.dvpp_set_pic_desc_data(self.dvpp_pic_desc,
                                         output_pic_mem)

        acl.media.dvpp_set_pic_desc_size(self.dvpp_pic_desc,
                                         output_pic_size)

        acl.media.dvpp_set_pic_desc_format(self.dvpp_pic_desc,
                                           self._format)
        print("[Vdec] create output pic desc success")

    def forward(self, output_pic_size, input_stream_size):
        self.frame_config = acl.media.vdec_create_frame_config()

        for i in range(self.rest_len):
            print("[Vdec] forward index:{}".format(i))
            self._set_input(input_stream_size)
            self._set_pic_output(output_pic_size)

            # vdec_send_frame
            ret = acl.media.vdec_send_frame(self.vdec_channel_desc,
                                            self.dvpp_stream_desc,
                                            self.dvpp_pic_desc,
                                            self.frame_config,
                                            None)
            check_ret("acl.media.vdec_send_frame", ret)
            print('[Vdec] vdec_send_frame stage success')

    def run(self, video_path):
        self.video_path, self.input_width, self.input_height, self.dtype = video_path

        timeout = 300
        cb_thread_id, ret = acl.util.start_thread(self._thread_func, [timeout])
        acl.rt.subscribe_report(cb_thread_id, self.stream)

        self.init_resource(cb_thread_id)

        output_pic_size = (self.input_width * self.input_height * 3) // 2
        self.input_stream_mem, input_stream_size = self. \
            _gen_input_dataset(self.video_path)

        self.forward(output_pic_size, input_stream_size)

        self._destroy_resource()

        self._vdec_exit = False
        ret = acl.util.stop_thread(cb_thread_id)
        check_ret("acl.util.stop_thread", ret)
        print("[Vdec] vdec finish!!!\n")

    def _destroy_resource(self):
        print("[Vdec] release resource:")
        ret = acl.media.dvpp_free(self.input_stream_mem)
        check_ret("acl.media.dvpp_free", ret)
        ret = acl.media.vdec_destroy_channel(self.vdec_channel_desc)
        check_ret("acl.media.vdec_destroy_channel", ret)

        ret = acl.media.vdec_destroy_channel_desc(self.vdec_channel_desc)
        check_ret("acl.media.vdec_destroy_channel_desc", ret)

        ret = acl.media.vdec_destroy_frame_config(self.frame_config)
        check_ret("acl.media.vdec_destroy_frame_config", ret)
        print("[Vdec] release resource success")
