"""
Yolov5 ONNX Exporter
Copyright 2022 Huawei Technologies Co., Ltd

Requirements:
    $ pip install -r requirements.txt  onnx onnx-simplifier onnxruntime  

Usage:
    $ python onnx_converter.py --weights yolov5s.pt 

CREATED:  2022-7-26 17:30:13
"""

import argparse, os, platform, sys, time, warnings,inspect,io ,torch, torch.nn as nn
from pathlib import Path
from typing import Optional
from utils.utils import *

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def export_onnx(model, im, file, opset, simplify, prefix=colorstr('ONNX:')):
    # YOLOv5 ONNX export
    try:
        import onnx
        print(f'\n{prefix} starting export with onnx {onnx.__version__}...')

        f = file[0][:-3] + '.onnx'
        print(f'Path: {f}')
        torch.onnx.export(
            model, 
            im,
            f,
            verbose=False,
            opset_version=opset,
            training=torch.onnx.TrainingMode.EVAL,
            do_constant_folding=True,
            input_names=['images'],
            output_names=['output'])

        # Checks
        model_onnx = onnx.load(f)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model

        # Metadata
        d = {'stride': int(max(model.stride)), 'names': model.names}
        for k, v in d.items():
            meta = model_onnx.metadata_props.add()
            meta.key, meta.value = k, str(v)
        onnx.save(model_onnx, f)

        # Simplify
        if simplify:
            try:
                import onnxsim
                print(f'{prefix} simplifying with onnx-simplifier {onnxsim.__version__}...')
                model_onnx, check = onnxsim.simplify(model_onnx)
                assert check, 'assert check failed'
                onnx.save(model_onnx, f)
            except Exception as e:
                print(f'{prefix} simplifier failure: {e}')
        print(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
        return f
    except Exception as e:
        print(f'{prefix} export failure: {e}')


@torch.no_grad()
def run(
        data=ROOT / 'data/coco128.yaml',  # 'dataset.yaml path'
        weights=ROOT / 'yolov5s.pt',  # weights path
        imgsz=(640, 640),  # image (height, width)
        batch_size=1,  # batch size
        device='cpu',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        inplace=False,  # set YOLOv5 Detect() inplace=True
        simplify=False,  # ONNX: simplify model
        opset=12,  # ONNX: opset version
):
    t = time.time()
    model = attempt_load(weights, device=device, inplace=True, fuse=True)  # load FP32 model        
    nc, names = model.nc, model.names  # number of classes, class names
    # Checks
    imgsz *= 2 if len(imgsz) == 1 else 1  # expand
    assert nc == len(names), f'Model class count {nc} != len(names) {len(names)}'

    # Input
    im = torch.zeros(batch_size, 3, *imgsz).to(device)  # image size(1,3,320,192) BCHW iDetection

    model.eval()  # training mode = no Detect() layer grid construction
    for k, m in model.named_modules():
        m.inplace = inplace
        m.onnx_dynamic = False
        m.export = True
    
    # Exports
    warnings.filterwarnings(action='ignore', category=torch.jit.TracerWarning)  # suppress TracerWarning
    f = export_onnx(model, im, weights, opset, simplify)
    # Finish
    print(f'\nExport complete ({time.time() - t:.2f}s)')
    print(f"\nResults saved!")
    return f  # return list of exported files/dirs

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640, 640], help='image (h, w)')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--inplace', action='store_true', help='set YOLOv5 Detect() inplace=True')
    parser.add_argument('--simplify', default= False ,action='store_true', help='ONNX: simplify model')
    parser.add_argument('--opset', type=int, default=12, help='ONNX: opset version. Currently, the ATC tool supports only opset_version=11.')
    opt = parser.parse_args()
    print_args(vars(opt))
    return opt

def main(opt):
    run(**vars(opt))
  
if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
