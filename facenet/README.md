Model URL
https://www.hiascend.com/en/software/modelzoo/detail/1/7548422b6b9c4a809114435f6b128bb7

Original Model Repository
https://github.com/davidsandberg/facenet

User Manual
https://support.huawei.com/enterprise/en/doc/EDOC1100206675/61c7aee9/sample-reference

Docker Image Download
https://ascendhub.huawei.com/#/detail/ascend-tensorflow

Docker command
docker run -it --rm -e ASCEND_VISIBLE_DEVICES=0 -v /home/a800/tianyu/tf_online_inference:/home/a800/tianyu/tf_online_inference ascendhub.huawei.com/public-ascendhub/ascend-tensorflow:21.0.2-ubuntu18.04

Python Command
python3 main.py --model_path ./models/facenet_tf.pb --input_tensor_name input:0 --output_tensor_name embeddings:0 --image_path ./facenet_data


