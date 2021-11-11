# Python Ascend Computing Language (ACL) Samples
 ## _N.B. Please convert the models based on the README of each sample!_ 
 ## _N.B. This repository is meant for sole learning purposes, not commercial use!_ 

#### Docker PyACL environment
Feel free to pull images from https://ascendhub.huawei.com/#/detail/infer-modelzoo. This image integrates ACL libs, model conversion tool (ATC), some python libraries

#### Sample Naming
Samples like acl_dvpp_* uses Atlas DVPP (Digital Vision Pre-Processing) hardware module for image and video decoding. Briefly, the acl_dvpp_* samples load the image as encoded raw bytes (Raw video frames in video streaming) and send those bytes to the DVPP module for decoding.

For samples acl_* without "dvpp", the image is read with opencv as BGR numpy array. The Model then takes this BGR image and returns whatever the models were trained to return.

#### Python dependencies
- Numpy
- Opencv
- Pillow
- Matplotlib

