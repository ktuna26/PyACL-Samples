# Model URL
Retrieve the PB file from the following link and place it under ./models  
https://www.hiascend.com/en/software/modelzoo/detail/1/b8b4f7e2dbd04120b082dbc44ab4a883

# Original Model Repository
https://github.com/davidsandberg/facenet

# User Manual
https://support.huawei.com/enterprise/en/doc/EDOC1100206675/61c7aee9/sample-reference

# Docker Image Download
Download the image version based on the CANN/driver/firmware version on your machine  
https://ascendhub.huawei.com/#/detail/ascend-tensorflow

# Docker command (example)
docker run -it --rm -e ASCEND_VISIBLE_DEVICES=0 -v /path/to/code:/path/to/code ascendhub.huawei.com/public-ascendhub/ascend-tensorflow:21.0.2-ubuntu18.04

# Python Command
python3 main.py --model_path ./models/facenet_tf.pb --input_tensor_name input:0 --output_tensor_name embeddings:0 --image_path ./facenet_data


