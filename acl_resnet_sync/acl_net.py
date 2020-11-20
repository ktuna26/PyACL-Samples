"""
Copyright (R) @huawei.com, all rights reserved
-*- coding:utf-8 -*-
CREATED:  2020-6-04 20:12:13
MODIFIED: 2020-6-17 14:04:45
"""
import argparse
import numpy as np
import struct
import acl
import os
from PIL import Image
from constant import ACL_MEM_MALLOC_HUGE_FIRST, \
    ACL_MEMCPY_HOST_TO_DEVICE, ACL_MEMCPY_DEVICE_TO_HOST, \
    ACL_ERROR_NONE, IMG_EXT, ACL_MEMCPY_DEVICE_TO_DEVICE

buffer_method = {
    "in": acl.mdl.get_input_size_by_index,
    "out": acl.mdl.get_output_size_by_index
    }


def check_ret(message, ret):
    if ret != ACL_ERROR_NONE:
        raise Exception("{} failed ret={}"
                        .format(message, ret))

class Net(object):
    def __init__(self, device_id, model_path):
        self.device_id = device_id      # int
        self.model_path = model_path    # string
        self.model_id = None            # pointer
        self.context = None             # pointer

        self.input_data = []
        self.output_data = []
        self.model_desc = None          # pointer when using
        self.load_input_dataset = None
        self.load_output_dataset = None

        self.model_dev_ptr_ = None
        self.weight_dev_ptr_ = None

        self.init_resource()

    def __del__(self):
        print("release source stage:")
        ret = acl.mdl.unload(self.model_id)
        check_ret("acl.mdl.unload", ret)
        if self.model_desc:
            acl.mdl.destroy_desc(self.model_desc)
            self.model_desc = None

        if self.model_dev_ptr_:
            acl.mdl.destroy_desc(self.model_dev_ptr_)
            self.model_dev_ptr_ = None

        if self.weight_dev_ptr_:
            acl.mdl.destroy_desc(self.weight_dev_ptr_)
            self.weight_dev_ptr_ = None

        while self.input_data:
            item = self.input_data.pop()
            ret = acl.rt.free(item["buffer"])
            check_ret("acl.rt.free", ret)

        while self.output_data:
            item = self.output_data.pop()
            ret = acl.rt.free(item["buffer"])
            check_ret("acl.rt.free", ret)

        if self.context:
            ret = acl.rt.destroy_context(self.context)
            check_ret("acl.rt.destroy_context", ret)
            self.context = None

        ret = acl.rt.reset_device(self.device_id)
        check_ret("acl.rt.reset_device", ret)
        ret = acl.finalize()
        check_ret("acl.finalize", ret)
        print('release source success')

    def init_resource(self):
        print("init resource stage:")
        ret = acl.init()
        check_ret("acl.init", ret)
        ret = acl.rt.set_device(self.device_id)
        check_ret("acl.rt.set_device", ret)

        self.context, ret = acl.rt.create_context(self.device_id)
        check_ret("acl.rt.create_context", ret)

        # load_model
        self.model_id, ret = acl.mdl.load_from_file(self.model_path)
        check_ret("acl.mdl.load_from_file", ret)
        print("model_id:{}".format(self.model_id))

        self.model_desc = acl.mdl.create_desc()
        self._get_model_info()
        print("init resource success")

    def _get_model_info(self,):
        ret = acl.mdl.get_desc(self.model_desc, self.model_id)
        check_ret("acl.mdl.get_desc", ret)
        input_size = acl.mdl.get_num_inputs(self.model_desc)
        output_size = acl.mdl.get_num_outputs(self.model_desc)
        self._gen_data_buffer(input_size, des="in")
        self._gen_data_buffer(output_size, des="out")

    def _gen_data_buffer(self, size, des):
        func = buffer_method[des]
        for i in range(size):
            # check temp_buffer dtype
            temp_buffer_size = func(self.model_desc, i)
            temp_buffer, ret = acl.rt.malloc(temp_buffer_size,
                                             ACL_MEM_MALLOC_HUGE_FIRST)
            check_ret("acl.rt.malloc", ret)

            if des == "in":
                self.input_data.append({"buffer": temp_buffer,
                                        "size": temp_buffer_size})
            elif des == "out":
                self.output_data.append({"buffer": temp_buffer,
                                         "size": temp_buffer_size})

    def _data_interaction(self, dataset, policy=ACL_MEMCPY_DEVICE_TO_DEVICE):
        
        temp_data_buffer = self.input_data \
            if len(dataset) != 0 \
            else self.output_data
#         print(temp_data_buffer, self.output_data)
        
#         if len(dataset) == 0 and policy == ACL_MEMCPY_DEVICE_TO_DEVICE:
#             print("In this IF")
#             for item in self.output_data:
#                 temp, ret = acl.rt.malloc(item["size"], ACL_MEM_MALLOC_NORMAL_ONLY)
#                 if ret != 0:
#                     raise Exception("can't malloc_host ret={}".format(ret))
#                 dataset.append({"size": item["size"], "buffer": temp})
#         return
        for i in range(len(temp_data_buffer)):
            item = temp_data_buffer[i]
            if len(dataset) != 0:
                ptr = acl.util.numpy_to_ptr(dataset[i])
                ret = acl.rt.memcpy(item["buffer"],
                                    item["size"],
                                    ptr,
                                    item["size"],
                                    policy)
                check_ret("acl.rt.memcpy", ret)

            else:
                dataset.append(item)
#                 ptr = dataset[i]["buffer"]
#                 ret = acl.rt.memcpy(ptr,
#                                     item["size"],
#                                     item["buffer"],
#                                     item["size"],
#                                     policy)
#                 check_ret("acl.rt.memcpy", ret)
#         print(dataset)

    def _destory_dataset_and_databuf(self, dataset):
        data_buf_num = acl.mdl.get_dataset_num_buffers(dataset)
        for i in range(data_buf_num):
            data_buf = acl.mdl.get_dataset_buffer(dataset, i)
            if data_buf is not None:
                ret = acl.destroy_data_buffer(data_buf)
                check_ret("acl.destroy_data_buffer", ret)
        ret = acl.mdl.destroy_dataset(dataset)
        check_ret("acl.mdl.destroy_dataset", ret)

    def _gen_dataset(self, type="input"):
        dataset = acl.mdl.create_dataset()

        temp_dataset = None
        if type == "in":
            self.load_input_dataset = dataset
            temp_dataset = self.input_data
        else:
            self.load_output_dataset = dataset
            temp_dataset = self.output_data

        for i in range(len(temp_dataset)):
            item = temp_dataset[i]
            data = acl.create_data_buffer(item["buffer"], item["size"])
            if data is None:
                ret = acl.destroy_data_buffer(dataset)
                check_ret("acl.destroy_data_buffer", ret)

            _, ret = acl.mdl.add_dataset_buffer(dataset, data)

            if ret != ACL_ERROR_NONE:
                ret = acl.destroy_data_buffer(dataset)
                check_ret("acl.destroy_data_buffer", ret)

    def _data_from_host_to_device(self, images):
        print("data interaction from host to device")
        # copy images to device
        self._data_interaction(images)
        # load input data into model
        self._gen_dataset("in")
        # load output data into model
        self._gen_dataset("out")
        print("data interaction from host to device success")

    def _data_from_device_to_host(self):
        print("data interaction from device to host")
        res = []
        # copy device to host
        self._data_interaction(res)
        result = self.get_result(res)
        self._print_result(result)
        print("data interaction from device to host success")

    def run(self, images):
        self._data_from_host_to_device(images)
        self.forward()
        self._data_from_device_to_host()

    def forward(self):
        print('execute stage:')
        ret = acl.mdl.execute(self.model_id,
                              self.load_input_dataset,
                              self.load_output_dataset)
        check_ret("acl.mdl.execute", ret)
        self._destroy_databuffer()
        print('execute stage success')

    def _print_result(self, result):
        st = struct.unpack("1000f", bytearray(result[0]))
        vals = np.array(st).flatten()
        top_k = vals.argsort()[-1:-6:-1]
        print("======== top5 inference results: =============")

        for n in top_k:
            print("[%d]: %f" % (n, vals[n]))

    def _destroy_databuffer(self, ):
        for dataset in [self.load_input_dataset, self.load_output_dataset]:
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
    parser.add_argument('--model_path', type=str,
                        default="./model/resnet50_aipp.om")
    parser.add_argument('--images_path', type=str, default="./data")
    args = parser.parse_args()
    print("Using device id:{}\nmodel path:{}\nimages path:{}"
          .format(args.device, args.model_path, args.images_path))

    net = Net(args.device, args.model_path)
    images_list = [os.path.join(args.images_path, img)
                   for img in os.listdir(args.images_path)
                   if os.path.splitext(img)[1] in IMG_EXT]

    for image in images_list:
        print("images:{}".format(image))
        transfer_pic(image)
        img = np.fromfile(image.replace(".jpg", ".bin"), dtype=np.byte)
        net.run([img])

    print("*****run finish******")
