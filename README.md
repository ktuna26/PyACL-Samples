# Python Ascend Computing Language (ACL) Samples
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

