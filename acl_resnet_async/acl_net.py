"""
Copyright (R) @huawei.com, all rights reserved
-*- coding:utf-8 -*-
CREATED:  2020-6-04 20:12:13
MODIFIED: 2020-6-06 14:04:45
"""
import struct
import numpy as np
import time
import os
import argparse
import acl

from PIL import Image
from constant import ACL_MEMCPY_HOST_TO_DEVICE, \
    ACL_MEMCPY_DEVICE_TO_HOST, ACL_MEM_MALLOC_NORMAL_ONLY, \
    ACL_ERROR_NONE, IMG_EXT


def check_ret(message, ret):
    if ret != ACL_ERROR_NONE:
        raise Exception("{} failed ret={}"
                        .format(message, ret))


class Net(object):
    def __init__(self, device_id,
                 execute_times,
                 callback_interval,
                 memory_pool,
                 model_path,
                 config_path=None,
                 ):

        self.device_id = device_id      # int
        self.model_path = model_path    # string
        self.config_path = config_path
        self.model_id = None            # pointer
        self.context = None             # pointer
        self.stream = None              # pointer
        self.excute_times = execute_times
        self.callback_interval = callback_interval
        self.is_callback = True if callback_interval else False
        self.memory_pool = memory_pool

        self.input_num = 0
        self.output_num = 0
        self.model_desc = None          # pointer when using
        self.dataset_list = []

        self.is_exist = False

        self.init_resource()

    def __del__(self):
        print('release source stage:')
        if self.model_id:
            ret = acl.mdl.unload(self.model_id)
            check_ret("acl.mdl.unload", ret)

        if self.model_desc:
            ret = acl.mdl.destroy_desc(self.model_desc)
            check_ret("acl.mdl.destroy_desc", ret)

        self._destory_dataset_and_databuf()

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
        print('release source stage success')

    def init_resource(self):
        print("init resource stage:")
        if isinstance(self.config_path, str) \
                and os.path.exists(self.config_path):
            ret = acl.init(self.config_path)
            check_ret("acl.init", ret)
        elif self.config_path is None:
            ret = acl.init()
            check_ret("acl.init", ret)
        ret = acl.rt.set_device(self.device_id)
        check_ret("acl.rt.set_device", ret)

        self.context, ret = acl.rt.create_context(self.device_id)
        check_ret("acl.rt.create_context", ret)

        self.stream, ret = acl.rt.create_stream()
        check_ret("acl.rt.create_stream", ret)
        self._get_model_info()
        print("init resource stage success")

    def _get_model_info(self,):
        self.model_id, ret = acl.mdl.load_from_file(self.model_path)
        check_ret("acl.mdl.load_from_file", ret)
        self.model_desc = acl.mdl.create_desc()
        ret = acl.mdl.get_desc(self.model_desc, self.model_id)
        check_ret("acl.mdl.get_desc", ret)
        self.input_num = acl.mdl.get_num_inputs(self.model_desc)
        self.output_num = acl.mdl.get_num_outputs(self.model_desc)

    def _load_input_data(self, images_data):
        img_ptr = acl.util.numpy_to_ptr(images_data)  # host ptr

        # memcopy host to device
        img_device, ret = acl.rt.malloc(
            images_data.size, ACL_MEM_MALLOC_NORMAL_ONLY)
        check_ret("acl.rt.malloc", ret)
        ret = acl.rt.memcpy(img_device, images_data.size, img_ptr,
                            images_data.size, ACL_MEMCPY_HOST_TO_DEVICE)
        check_ret("acl.rt.memcpy", ret)

        # create dataset in device
        img_dataset = acl.mdl.create_dataset()
        img_data_buffer = acl.create_data_buffer(img_device, images_data.size)
        if img_data_buffer is None:
            raise Exception("can't create data buffer, create input failed!!!")

        _, ret = acl.mdl.add_dataset_buffer(img_dataset, img_data_buffer)
        if ret != ACL_ERROR_NONE:
            ret = acl.destroy_data_buffer(img_dataset)
            check_ret("acl.destroy_data_buffer", ret)
        return img_dataset

    def _load_output_data(self):
        output_data = acl.mdl.create_dataset()
        for i in range(self.output_num):
            # check temp_buffer dtype
            temp_buffer_size = acl.mdl.get_output_size_by_index(
                self.model_desc, i)
            temp_buffer, ret = acl.rt.malloc(
                temp_buffer_size, ACL_MEM_MALLOC_NORMAL_ONLY)
            check_ret("acl.rt.malloc", ret)

            data_buf = acl.create_data_buffer(temp_buffer, temp_buffer_size)
            _, ret = acl.mdl.add_dataset_buffer(output_data, data_buf)
            if ret != ACL_ERROR_NONE:
                acl.destroy_data_buffer(output_data)
                check_ret("acl.destroy_data_buffer", ret)
        return output_data

    def _data_interaction(self, images_dataset_list):
        print("data interaction from host to device")
        for idx in range(self.memory_pool):
            img_idx = idx % len(images_dataset_list)
            img_input = self._load_input_data(images_dataset_list[img_idx])
            infer_ouput = self._load_output_data()
            self.dataset_list.append([img_input, infer_ouput])
        print("data interaction from host to device success")

    def _destory_dataset_and_databuf(self, ):
        while self.dataset_list:
            dataset = self.dataset_list.pop()
            for temp in dataset:
                num_temp = acl.mdl.get_dataset_num_buffers(temp)
                for i in range(num_temp):
                    data_buf_temp = acl.mdl.get_dataset_buffer(temp, i)
                    if data_buf_temp:
                        ret = acl.destroy_data_buffer(data_buf_temp)
                        check_ret("acl.destroy_data_buffer", ret)
                ret = acl.mdl.destroy_dataset(temp)
                check_ret("acl.mdl.destroy_dataset", ret)

    def _process_callback(self, args_list):
        context, time_out = args_list

        acl.rt.set_context(context)
        while self.callback_interval:
            ret = acl.rt.process_report(time_out)
            check_ret("acl.rt.process_report", ret)
            if self.is_exist:
                print("exist acl.rt.process_report")
                break

    def run(self, images):
        if not isinstance(images, list):
            raise Exception("images isn't list")

        # copy images to device
        self._data_interaction(images)

        # thread
        tid, ret = acl.util.start_thread(self._process_callback,
                                         [self.context, 50])
        check_ret("acl.util.start_thread", ret)

        ret = acl.rt.subscribe_report(tid, self.stream)
        check_ret("acl.rt.subscribe_report", ret)

        self.forward()
        ret = acl.rt.synchronize_stream(self.stream)
        check_ret("acl.rt.synchronize_stream", ret)
        ret = acl.rt.unsubscribe_report(tid, self.stream)
        check_ret("acl.rt.unsubscribe_report", ret)
        self.is_exist = True
        ret = acl.util.stop_thread(tid)
        check_ret("acl.util.stop_thread", ret)

    def _get_callback(self, idx):
        if (idx + 1) % self.callback_interval == 0:
            ret = acl.rt.launch_callback(self.callback_func,
                                         self.excute_dataset,
                                         1,
                                         self.stream)
            check_ret("acl.rt.launch_callback", ret)
            self.dataset_list.extend(self.excute_dataset)
            self.excute_dataset = []

    def forward(self):
        print('execute stage:')
        self.excute_dataset = []
        for idx in range(self.excute_times):
            img_data, infer_output = self.dataset_list.pop(0)
            ret = acl.mdl.execute_async(self.model_id,
                                        img_data,
                                        infer_output,
                                        self.stream)
            check_ret("acl.mdl.execute_async", ret)

            if self.is_callback:
                self.excute_dataset.append([img_data, infer_output])
                self._get_callback(idx)
        print('execute stage success')

    def callback_func(self, delete_list):
        print('callback func stage:')
        for temp in delete_list:
            _, infer_output = temp
            # device to host
            num = acl.mdl.get_dataset_num_buffers(infer_output)
            for i in range(num):
                temp_output_buf = acl.mdl.get_dataset_buffer(infer_output, i)

                infer_output_ptr = acl.get_data_buffer_addr(temp_output_buf)
                infer_output_size = acl.get_data_buffer_size(temp_output_buf)

                output_host, ret = acl.rt.malloc_host(infer_output_size)
                check_ret("acl.rt.malloc_host", ret)
                ret = acl.rt.memcpy(output_host,
                                    infer_output_size,
                                    infer_output_ptr,
                                    infer_output_size,
                                    ACL_MEMCPY_DEVICE_TO_HOST)
                check_ret("acl.rt.memcpy", ret)
                output_host_dict = [
                    {"buffer": output_host, "size": infer_output_size}]
                result = self.get_result(output_host_dict)
                st = struct.unpack("1000f", bytearray(result[0]))
                vals = np.array(st).flatten()
                top_k = vals.argsort()[-1:-6:-1]
                print("\n======== top5 inference results: =============")
                for n in top_k:
                    print("[%d]: %f" % (n, vals[n]))
                # acl.rt.free_host(output_host)
        print('callback func stage success')

    def get_result(self, output_data):
        dataset = []
        for i in range(len(output_data)):
            temp = output_data[i]
            size = temp["size"]
            ptr = temp["buffer"]
            data = acl.util.ptr_to_numpy(ptr, (size,), 1)
            dataset.append(data)
        return dataset


def transfer_pic(input_path):
    input_path = os.path.abspath(input_path)
    im = Image.open(input_path)
    im = im.resize((256, 256))
    # hwc
    img = np.array(im)
    height = img.shape[0]
    width = img.shape[1]
    h_off = int((height - 224) / 2)
    w_off = int((width - 224) / 2)
    crop_img = img[h_off:height - h_off, w_off:width - w_off, :]
    # rgb to bgr
    img = crop_img[:, :, ::-1]
    shape = img.shape
    img = img.astype("float16")
    img[:, :, 0] -= 104
    img[:, :, 1] -= 117
    img[:, :, 2] -= 123
    img = img.reshape([1] + list(shape))
    result = img.transpose([0, 3, 1, 2])
    outputName = input_path.split('.')[0] + ".bin"
    result.tofile(outputName)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--execute_times', type=int, default=10)
    parser.add_argument('--callback_interval', type=int, default=1)
    parser.add_argument('--mem_pools', type=int, default=10)
    parser.add_argument('--model_path', type=str,
                        default="./model/resnet50.om")
    parser.add_argument('--images_path', type=str, default="./data")
    args = parser.parse_args()
    print("Using device id:{}\n"
          "execute_times:{} \n"
          "callback_interval:{}\n"
          "mem_pools:{}\n"
          "model path:{}\n"
          "images path:{}".format(args.device, args.execute_times,
                                  args.callback_interval, args.mem_pools,
                                  args.model_path, args.images_path))
    net = Net(args.device, args.execute_times,
              args.callback_interval, args.mem_pools,
              args.model_path)
    images_list = [os.path.join(args.images_path, img)
                   for img in os.listdir(args.images_path)
                   if os.path.splitext(img)[1] in IMG_EXT]

    data_list = []
    for image in images_list:
        transfer_pic(image)
        dst_im = np.fromfile(image.replace(".jpg", ".bin"), dtype=np.byte)
        data_list.append(dst_im)

    net.run(data_list)
    end = time.time()
    print("run finish!!!")
