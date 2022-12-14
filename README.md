# Python Ascend Computing Language (ACL) Samples

Ascend Computing Language (AscendCL) provides a collection of C language APIs for users to develop deep neural network apps for object recognition and image classification, ranging from device management, context management, stream management, memory management, to model loading and execution, operator loading and execution, and media data processing. You can call AscendCL APIs through a third-party framework to utilize the compute capability of the Ascend AI Processor, or encapsulate AscendCL into third-party libraries, to provide the runtime and resource management capabilities of the Ascend AI Processor.

This repository provides a wide range of samples developed based on AscendCL APIs. When developing your own samples, feel free to refer to the existing samples in this repository.

**Note:** _This repository is meant for sole learning purposes, not commercial use!_ 

## Directory Structure
| directory | description |
|---|---|
| [Caffe](./Caffe) | PyACL sampels for Caffe models |
| [Common](./Common) | Common files or libs for projets |
| [MindSpore](./MindSpore) | PyACL sampels for MindSpore models |
| [PyTorch](./PyTorch) | PyACL sampels for PyTorch models |
| [Tensorflow](./Tensorflow) | PyACL sampels for Tensorflow models |

#### Docker PyACL environment
Feel free to pull images from https://ascendhub.huawei.com/#/detail/infer-modelzoo. This image integrates ACL libs, model conversion tool (ATC), some python libraries

## Sample Deployment

Deploy and run a sample by referring to the corresponding README file explaining model conversion steps.

**Note:** If you don't want to struggle to prepare development environment, you can use ready to use docker images from AscendHub. So, feel free to pull images from https://ascendhub.huawei.com/#/detail/infer-modelzoo. This image integrates ACL libs, model conversion tool (ATC), some python libraries.

## Sample Naming
Samples like acl_dvpp_* uses Atlas DVPP (Digital Vision Pre-Processing) hardware module for image and video decoding. Briefly, the acl_dvpp_* samples load the image as encoded raw bytes (Raw video frames in video streaming) and send those bytes to the DVPP module for decoding.

For samples acl_* without "dvpp", the image is read with opencv as BGR numpy array. The Model then takes this BGR image and returns whatever the models were trained to return.

## Python dependencies
- Numpy
- Opencv
- Pillow
- Matplotlib
- Jupyter Notebook

## Documentation

Obtain related documentation at [Ascend Documentation](https://www.hiascend.com/document).

## Community

Get support from our developer community. Find answers and talk to other developers in our forum.

Ascend website: https://www.hiascend.com

Ascend forum: https://forum.huawei.com/enterprise/en/forum-100504.html

## License
[MIT](LICENSE)

</br></br>
<p align="center">
<img src="https://r.huaweistatic.com/s/ascendstatic/lst/header/header-logo.png" align="center"/>
</p>