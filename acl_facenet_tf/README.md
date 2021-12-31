# Tensorflow FaceNet
Please open the `jupyter-notebook` for a quick demo | [Pretrained Model](https://gitee.com/link?target=https%3A%2F%2Fmodelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com%2F003_Atc_Models%2Fmodelzoo%2FOfficial%2Fcv%2FFacenet_for_ACL.zip) [Paper](https://arxiv.org/abs/1503.03832) [Original Github Repository](https://github.com/davidsandberg/facenet)

## Overview
FaceNet is a general-purpose system that can be used for face verification (is it the same person?), recognition (who is this person?), and cluster (how to find similar people?). FaceNet uses a convolutional neural network to map images into Euclidean space. The spatial distance is directly related to the image similarity. The spatial distance between different images of the same person is small, and the spatial distance between images of different persons is large. As long as the mapping is determined, face recognition becomes simple. FaceNet directly uses the loss function of the triplets-based LMNN (large margin nearest neighbor) to train the neural network. The network directly outputs a 512-dimensional vector space. The triples we selected contain two matched face thumbnails and one unmatched face thumbnail. The objective of the loss function is to distinguish positive and negative classes by distance boundary.

## Getting started
Install dependencies;
- opencv-python>=3.4.2
- numpy

```
pip install -r requirements.txt
```
And then download the `PT` file of from the link.

### TF model -> Ascend om format
#### PB -> OM
And then use in the same directory atc tool to convert `ONNX ` file to `OM` file as as follows.
```bash
atc --model=facenet_tf.pb \
      --framework=3 \
      --output=facenet_tf  \
      --soc_version=Ascend310 \
      --input_shape="input:1,160,160,3"
```

Finaly, open `jupyter-notebook` and run the code for demo