"""
Copyright (R) @huawei.com, all rights reserved
-*- coding:utf-8 -*-
CREATED:  2020-6-04 20:12:13
MODIFIED: 2020-6-28 14:04:45
"""
import numpy as np
import os
import argparse
import acl
from PIL import Image
from constant import ACL_MEMCPY_HOST_TO_DEVICE, ACL_ERROR_NONE, \
        IMG_EXT, ACL_MEMCPY_DEVICE_TO_DEVICE
from acl_model import Model, check_ret
from acl_dvpp import Dvpp
from acl_op import SingleOp


class Sample(object):
    def __init__(self,
                 device_id,
                 model_path,
                 model_input_width,
                 model_input_height):
        self.device_id = device_id      # int
        self.model_path = model_path    # string
        self.model_id = None            # pointer
        self.context = None             # pointer

        self.input_data = None
        self.output_data = None
        self.model_desc = None          # pointer when using
        self.load_input_dataset = None
        self.load_output_dataset = None
        self.init_resource()

        self._model_input_width = model_input_width
        self._model_input_height = model_input_height

        self.model_process = Model(self.context,
                                   self.stream,
                                   model_path)

        self.dvpp_process = Dvpp(self.stream,
                                 model_input_width,
                                 model_input_height)

        self.sing_op = SingleOp(self.stream)

    def __del__(self):
        if self.model_process:
            del self.model_process

        if self.dvpp_process:
            del self.dvpp_process

        if self.sing_op:
            del self.sing_op

        if self.stream:
            acl.rt.destroy_stream(self.stream)

        if self.context:
            acl.rt.destroy_context(self.context)
        acl.rt.reset_device(self.device_id)
        acl.finalize()
        print("[Sample] class Samle release source success")

    def init_resource(self):
        print("[Sample] init resource stage:")
        acl.init()
        ret = acl.rt.set_device(self.device_id)
        check_ret("acl.rt.set_device", ret)

        self.context, ret = acl.rt.create_context(self.device_id)
        check_ret("acl.rt.create_context", ret)

        self.stream, ret = acl.rt.create_stream()
        check_ret("acl.rt.create_stream", ret)
        print("[Sample] init resource stage success")

    def _transfer_to_devicce(self, img_path, dtype=np.uint8):
        img = np.fromfile(img_path, dtype=dtype)
        img_ptr = acl.util.numpy_to_ptr(img)
        img_buffer_size = img.itemsize * img.size
        
        img_device, ret = acl.media.dvpp_malloc(img_buffer_size)
        check_ret("acl.media.dvpp_malloc", ret)
        ret = acl.rt.memcpy(img_device,
                            img_buffer_size,
                            img_ptr,
                            img_buffer_size,
                            ACL_MEMCPY_DEVICE_TO_DEVICE)
        check_ret("acl.rt.memcpy", ret)

        # ret = acl.rt.free_host(img_ptr)
        # check_ret("acl.rt.free_host", ret)
        return img_device, img_buffer_size

    def forward(self, img_dict):
        img_path, image_dtype = img_dict["path"], img_dict["dtype"]
        # copy images to device
        im = Image.open(img_path)
        width, height = im.size
        print("[Sample] width:{} height:{}".format(width, height))
        print("[Sample] image:{}".format(img_path))
        img_device, img_buffer_size = self._transfer_to_devicce(img_path, img_dict["dtype"])

        # decode and resize
        dvpp_output_buffer, dvpp_output_size = \
            self.dvpp_process.run(img_device,
                                  img_buffer_size,
                                  width,
                                  height)

        print("dvpp_output_size", dvpp_output_size)
        output_data = self.model_process.run(
            dvpp_output_buffer,
            dvpp_output_size)
        
        return output_data
        print("output_data type", type(output_data))
        py_output_data = acl.util.ptr_to_numpy(output_data, (1, 1000, 1, 1), 11)

        print("inference output tensor size", acl.get_tensor_desc_size(output_data))
        print(py_output_data.shape)
        print(np.argmax(py_output_data.flatten()))
        #self.sing_op.run(output_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--model_path', type=str,
                        default="./model/resnet50_aipp.om")
    parser.add_argument('--model_input_width', type=int, default=224)
    parser.add_argument('--model_input_weight', type=int, default=224)
    parser.add_argument('--images_path', type=str, default="./data")
    args = parser.parse_args()
    print("Using device id:{}\nmodel path:{}\nimages path:{}"
          .format(args.device, args.model_path, args.images_path))

    sample = Sample(args.device,
                    args.model_path,
                    args.model_input_width,
                    args.model_input_weight)
    images_list = [os.path.join(args.images_path, img)
                   for img in os.listdir(args.images_path)
                   if os.path.splitext(img)[1] in IMG_EXT]

    for image in images_list:
        img_dict = {"path": image, "dtype": np.uint8}
        sample.forward(img_dict)
