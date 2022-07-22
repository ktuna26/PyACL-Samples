# PyTorch YoloV5 Object Detection
Please open the `jupyter-notebook` for a quick demo | [Pretrained Model](https://github.com/ultralytics/yolov5/releases/tag/v6.1) | [Original Github Repository (v6.1)](https://github.com/ultralytics/yolov5/tree/v6.1)

## PT model -> ONNX format -> Ascend om format
### PT -> ONNX
Use this step to convert  **`yolov5s.pt`**  to  **`yolov5s.onnx`**. 
Use in the original repository the `export.py` the script to convert `PT` file to `ONNX ` file.
```
git clone https://github.com/ultralytics/yolov5/tree/v6.1 
``` 
We recomend to use python virtual environment for PT->ONNX conversion.

- Example for python virtual environment: 
```bash
python -m venv pt2onnxExport

source pt2onnxExport/bin/activate
```

- Example for conda virtual environment:

```bash
conda create -n ENV_NAME

conda activate ENV_NAME
```
- Use specific version of torch and torchvision packages;
```bash
pip install -r requirements.txt
pip install torch==1.10.2 && pip install torchvision==0.11.3
```

```
python export.py --weights yolov5s.pt --include onnx
```

(If you got conversion error while converting PT->ONNX you should check your `torch` version and `torchvision` version)

### Remove a few operators in the ONNX file (Optional)
The  **Slice** and  **Transpose** operators will slow down the model inference significantly. Use `./model/modify_yolov5.py` script in this repo to remove the impact of these operators.  

### ONNX -> OM

```bash
cd ./model
atc --model=yolov5s_sim_t.onnx \
    --framework=5 \
    --output=yolov5s_sim_aipp \
    --input_format=NCHW \
    --log=error \
    --soc_version=Ascend310 \ # For different chip architectures change soc_version variable (Ascend310/Ascend910)
    --input_shape="images:1,3,640,640" \
    --enable_small_channel=1 \
    --output_type=FP16 \
    --insert_op_conf=aipp.cfg
```

## Benchmark
The benchmark is conducted on a Huawei Atlas 800 3010 X86 inference server (Ascend310) and Huawei Atlas 800 9010 x86 training server (Ascend910) with CANN 21.0.4 and models from; 
Yolov5 v2.0 = https://github.com/ultralytics/yolov5/releases/tag/v2.0 
Yolov5 v6.1 = https://github.com/ultralytics/yolov5/releases/tag/v6.1

The latency only covers the model inference (graph run),  **EXCLUDING**  YOLO post-processing

<img src="/data/9010_6vs2.png" width=650>
<img src="/data/A800_3010_YOLO.png" width=650>

##### Huawei Atlas 800 9010 X86 training server (Ascend910)
| Model   | Latency (ms) (v2.0) |Latency (ms) (v6.1) |
|---------|--------------|-------------|
| yolov5s | 8.26         |3.22|
| yolov5m | 13.47        |3.76|
| yolov5l | 22.86        |3.99|
| yolov5x | 35.96        |4.31|

##### Huawei Atlas 800 3010 X86 inference server (Ascend310)
| Model   | Latency (ms) (v2.0) |Latency (ms) (v6.1) |
|---------|--------------|-------------|
| yolov5s | 8.26         |4.51|
| yolov5m | 13.47        |6.36|
| yolov5l | 22.86        |7.23|
| yolov5x | 35.96        |8.77|

### Jupyter Notebook Example Output

<img src="/data/example.png" width=650>