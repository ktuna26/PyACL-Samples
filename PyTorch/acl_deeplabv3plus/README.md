# PyTorch Deeplabv3 Plus

Please open the `jupyter-notebook` for a quick demo | [Pretrained Model](https://www.hiascend.com/zh/software/modelzoo/models/detail/1/76f4e072a489484f98073591b912ad16/1) |[Original Github Repository](https://github.com/open-mmlab/mmsegmentation)

## Semantic Segmentation Sample

Function: uses the DeepLabv3 model to perform semantic segmentation on the input image and print the result on the output image.  
Input: JPG images.   
Output: JPG images with inference results

### Sample Preparation

**1)** Obtain the source model required by the application.
   
    | **Model** | **Description** | **How to Obtain** |
    |---|---|---|
    | ATC DeepLabV3+ (FP16)| An inference model for semantic segmentation  | Download pretrained model [ModelZoo-Ascend Community](https://www.hiascend.com/zh/software/modelzoo/models/detail/1/76f4e072a489484f98073591b912ad16/1) |

**Note:** Change the folder name `Deeplabv3+` to `Deeplabv3plus` due to fixing atc conversion error.

**2)** Convert your model.
 
##### ONNX format -> Ascend om format

```bash
# ATC Model conversion Ascend910
atc --model=deeplabv3_sim_bs1.onnx --framework=5 --output_type=FP16 --output=deeplabv3plus513_910 --input_shape="actual_input_1:1,3,513,513" --soc_version=Ascend910

# ATC Model conversion Ascend310
atc --model=deeplabv3_sim_bs1.onnx --framework=5 --output_type=FP16 --output=deeplabv3plus513_310 --input_shape="actual_input_1:1,3,513,513" --soc_version=Ascend310
```   

3. Obtain the test image required by the sample.
```bash
cd /data
wget https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/deeplabv3/test_image/test.jpg
```  

### Sample Running

Finaly, open jupyter-notebook and run the code for demo

### Jupyter Notebook Example Output

<img src="./out/test.jpg" width=650>