import torch
from blazeface import BlazeFace
from collections import OrderedDict

############### add blazeface.py file ###############
from collections import OrderedDict

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict
############### add blazeface.py file ###############

# load net
net = BlazeFace(back_model=True).to(torch.device("cpu"))
net.load_weights("blazefaceback.pth")

# load data
img = torch.zeros((1, 3, 256, 256)) # BCHW

# trace export
torch.onnx.export(net, img, 'blazefaceback.onnx', export_params=True, verbose=True, opset_version=11)
# torch.onnx.export(net, img, 'blazefaceback.onnx', export_params=True, verbose=True)