"""
Copyright (R) @huawei.com, all rights reserved
-*- coding:utf-8 -*-
CREATED:  2021-10-13 13:12:13
MODIFIED: 2020-10-14 15:04:45

First, please install requirements.txt

Usage python onnx_exporter.py --model [MODEL_NAME] --output [OUTPUT_NAME]

"""

import os,wget
os.environ['TF_KERAS'] = '1'
import onnx, keras2onnx
from models import arcface_model, vgg_face_model, facenet_model, deepface_model, openface_model, deepid_model
import argparse

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights',type=str, help='model weight path ')
    parser.add_argument('--model', type=str, help='model names are: arcface, vggface, facenet, deepface, openface, deepid')
    parser.add_argument('--output', type=str, help='output onnx model name')

    opt = parser.parse_args()
    return opt


def weight_download(model,model_name):
	cwd = os.getcwd()

	model_dict = {'arcface': 'https://github.com/serengil/deepface_models/releases/download/v1.0/arcface_weights.h5',
				  'vggface': 'https://github.com/serengil/deepface_models/releases/download/v1.0/vgg_face_weights.h5',
				  'facenet': 'https://github.com/serengil/deepface_models/releases/download/v1.0/facenet_weights.h5',
				  'deepface': 'https://github.com/swghosh/DeepFace/releases/download/weights-vggface2-2d-aligned/VGGFace2_DeepFace_weights_val-0.9034.h5.zip',
				  'openface': 'https://github.com/serengil/deepface_models/releases/download/v1.0/openface_weights.h5',
				  'deepid': 'https://github.com/serengil/deepface_models/releases/download/v1.0/deepid_keras_weights.h5'}
	
	filename = wget.download(model_dict[model_name])

	if model_name == 'deepface':
		import zipfile
		with zipfile.ZipFile(f"{cwd}/VGGFace2_DeepFace_weights_val-0.9034.h5.zip","r") as zip_ref:
			zip_ref.extractall(cwd) 
		os.remove(f"{cwd}/VGGFace2_DeepFace_weights_val-0.9034.h5.zip")
		filename = "VGGFace2_DeepFace_weights_val-0.9034.h5"
	
	model.load_weights(f"{cwd}/{filename}")
	os.remove(f'{cwd}/{filename}')
	return model
	

def models(model_name,weight):
	if model_name == 'arcface':
		model = arcface_model.Arcface()
		model = weight_download(model,model_name)
		return model

	elif model_name == 'vggface':
		model = vgg_face_model.VGGFace()
		model = weight_download(model,model_name)

		from keras.models import Model
		vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)

		return vgg_face_descriptor
	
	elif model_name == 'facenet':
		model = facenet_model.facenet()
		model = weight_download(model,model_name)
		return model

	elif model_name == 'deepface':
		model = deepface_model.Deepface()
		model = weight_download(model,model_name)

		from tensorflow.keras.models import Model
		deepface = Model(inputs=model.layers[0].input, outputs=model.layers[-3].output)

		return deepface

	elif model_name == 'openface':
		model = openface_model.OpenFace()
		model = weight_download(model,model_name)
		return model

	elif model_name == 'deepid':
		model = deepid_model.DeepID()
		model = weight_download(model,model_name)
		return model
		
	else:
		raise NameError(f'{model_name} is not in the model list or written model name wrong!')


def onnx_extraction(model,output_name):
	onnx_model = keras2onnx.convert_keras(model, model.name, target_opset=11)
	onnx.save_model(onnx_model, f"{output_name}.onnx")


def run(weights,model,output):
	selected_model = models(model,weights)
	onnx_extraction(selected_model,output)


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)