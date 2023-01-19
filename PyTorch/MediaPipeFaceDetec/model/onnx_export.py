"""
Copyright 2022 Huawei Technologies Co., Ltd

CREATED:  2022-10-04 13:12:13
MODIFIED: 2022-12-13 22:48:45
"""

# -*- coding:utf-8 -*-
import torch
from blazeface import BlazeFace


# load net
net = BlazeFace(back_model=True).to(torch.device("cpu"))
net.load_weights("blazefaceback.pth")

# load data
img = torch.zeros((1, 3, 256, 256)) # BCHW

# trace export
torch.onnx.export(net, img, 'blazefaceback.onnx', export_params=True, verbose=True, opset_version=11)