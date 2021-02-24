# Python Ascend Computing Language (ACL) Samples
 ## _N.B. Please convert the models based on the README of each sample!_ 

#### Docker PyACL environment
Feel free to `docker pull tianyuzhouhw/atlas_dev_env:amd64.v20.2`. This image integrates ACL libs 20.2, model conversion tool (ATC), jupyter notebook and a bunch of python dependencies
-  **On Atlas 500** : docker run -it --rm -p 8888:8888  --device=/dev/davinci0 --device=/dev/davinci_manager --device=/dev/hisi_hdc --device /dev/devmm_svm  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi  -v /home/data/miniD/driver/lib64:/home/data/miniD/driver/lib64 tianyuzhouhw/atlas_dev_env:aarch64.v20.2

-  **On a non-Atlas device** : docker run -it --rm tianyuzhouhw/atlas_dev_env:aarch64.v20.2 /bin/bash

#### Installation of pyACL
https://support.huaweicloud.com/intl/en-us/asdevg-python-cann/atlaspython_01_0006.html
#### ACL Overview
https://support.huaweicloud.com/intl/en-us/asdevg-python-cann/atlaspython_01_0008.html
#### API manual
https://support.huaweicloud.com/intl/en-us/asdevg-python-cann/atlaspyapi_07_0002.html  
Search for the specification of one acl function using the search bar on the left.

#### Sample Naming
Samples like acl_dvpp_* uses Atlas DVPP (Digital Vision Pre-Processing) hardware module for image and video decoding. Briefly, the acl_dvpp_* samples load the image as encoded raw bytes (Raw video frames in video streaming) and send those bytes to the DVPP module for decoding.

For samples acl_* without "dvpp", the image is read with opencv as BGR numpy array. The Model then takes this BGR image and returns whatever the models were trained to return.

#### Python dependencies
- Numpy
- Opencv
- Pillow
- Matplotlib

