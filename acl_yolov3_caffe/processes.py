import acl, time, cv2
import sys
sys.path.append('../acllite')
from acllite_model import AclLiteModel
from acllite_resource import AclLiteResource
import acl, cv2, struct, time
import numpy as np
from PIL import Image, ImageDraw
import os


labels = ["person",
        "bicycle", "car", "motorbike", "aeroplane",
        "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
        "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
        "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
        "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
        "pizza", "donut", "cake", "chair", "sofa", "potted plant", "bed", "dining table",
        "toilet", "TV monitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush"]

MODEL_WIDTH = 416
MODEL_HEIGHT = 416

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0)]

def construct_image_info():
    """construct image info"""
    image_info = np.array([MODEL_WIDTH, MODEL_HEIGHT, 
                           MODEL_WIDTH, MODEL_HEIGHT], 
                           dtype = np.float32) 
    return image_info

def get_model_info(model):
    i = 0
    o = 0
    print(f'=================================\n\t\033[1mInput Dimensions\033[0m\n=================================')
    while i>=0:
        try:
            input_dims = acl.mdl.get_input_dims(model._model_desc, i)

            print(f"\033[32mName\033[0m: {input_dims[0]['name']}\n\033[36mDimensions\033[0m: {input_dims[0]['dims']}\n---------------------------------")
            i += 1
            acl.mdl.get_input_dims(model._model_desc, i)[0]['dims']
        except: i = -1
    print('='*33)
    print(f'\n\n=================================\n\t\033[1mOutput Dimensions\033[0m\n=================================')
    while o>=0:
        try:
            output_dims = acl.mdl.get_output_dims(model._model_desc, o)
            print(f"\033[32mName\033[0m: {output_dims[0]['name']}\n\033[36mDimensions\033[0m: {output_dims[0]['dims']}\n---------------------------------")
            o += 1
            acl.mdl.get_output_dims(model._model_desc, o)[0]['dims']
        except: o = -1
    print('='*33)

def get_sizes(model_desc):
    output_size = acl.mdl.get_num_outputs(model_desc)
    input_size = acl.mdl.get_num_inputs(model_desc)

    print("model input size", input_size)
    for i in range(input_size):
        print("input ", i)
        print("model input dims", acl.mdl.get_input_dims(model_desc, i))
        print("model input datatype", acl.mdl.get_input_data_type(model_desc, i))
        model_input_height, model_input_width = (i for i in acl.mdl.get_input_dims(model_desc, i)[0]['dims'][1:3])

    print("=" * 50)
    print("model output size", output_size)
    for i in range(output_size):
        print("output ", i)
        print("model output dims", acl.mdl.get_output_dims(model_desc, i))
        print("model output datatype", acl.mdl.get_output_data_type(model_desc, i))
        model_output_height, model_output_width = acl.mdl.get_output_dims(model_desc, i)[0]['dims']

    print("=" * 50)
    print("[Model] class Model init resource stage success")
    return model_input_height,model_input_width,model_output_height,model_output_width

def preprocessing(img,model_desc):
    #model_input_height, model_input_width ,_ ,_ = get_sizes(model_desc)
    image_height, image_width = img.shape[:2]
    img_resized = letterbox_resize(img, 416, 416)[:, :, ::-1]
    # img_resized = self.resize_image(img, (self.model_input_width, self.model_input_height))[:, :, ::-1]
    # img_resized = (img_resized / 255).astype(np.float32).transpose([2, 0, 1])
    img_resized = np.ascontiguousarray(img_resized)
    print("img_resized shape", img_resized.shape)
    return img_resized

def letterbox_resize(img, new_width, new_height, interp=0):
    '''
    Letterbox resize. keep the original aspect ratio in the resized image.
    '''
    ori_height, ori_width = img.shape[:2]
    resize_ratio = min(new_width / ori_width, new_height / ori_height)
    resize_w = int(resize_ratio * ori_width)
    resize_h = int(resize_ratio * ori_height)

    # img = cv2.resize(img, (resize_w, resize_h), interpolation=interp)
    # img = cv2.resize(img, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
    img = cv2.resize(img, (resize_w, resize_h), interpolation=cv2.INTER_NEAREST)

    image_padded = np.full((new_height, new_width, 3), 128, np.uint8)

    dw = int((new_width - resize_w) / 2)
    dh = int((new_height - resize_h) / 2)

    image_padded[dh: resize_h + dh, dw: resize_w + dw, :] = img
    # return image_padded, resize_ratio, dw, dh
    return image_padded

def post_process(infer_output, bgr_img, image_file):
    """postprocess"""
    print("post process")
    box_num = infer_output[1][0, 0] # Kac adet bounding box olacagini return eder
    box_info = infer_output[0].flatten() # ic ice listeyi temizler
    
    # Modelin en ve boy ile orantisi
    print(f'image shape = {bgr_img.shape}')
    scalex = bgr_img.shape[1] / MODEL_WIDTH
    print(f'scalex : {scalex}')
    scaley = bgr_img.shape[0] / MODEL_HEIGHT
    print(f'scaley : {scaley}')
    
    '''if scalex > scaley:
        scaley =  scalex
        print(f'new scalex : {scalex}')
        print(f'new scaley : {scaley}')
    ''' 
    if not os.path.exists('./out'):
        os.makedirs('./out')
    output_path = os.path.join("./out", os.path.basename(image_file))
    print(f'output path: {output_path}')
    print("image file = ", image_file)
    
    print(f'box_num: {box_num}')
    for n in range(int(box_num)):
        ids = int(box_info[5 * int(box_num) + n]) # Class ID'si box_num'dan cekilir
        label = labels[ids] # Class Labeli
        score = box_info[4 * int(box_num)+n] # Confidence (?)
        #print(f'\n\n=========\n Box Info = {box_info} \n=========\n\n')
        ##############################################################################
        top_left_x = box_info[0 * int(box_num) + n] * scalex
        #print(f'top_left_x = box_info[{0 * int(box_num) + n}] * {scalex} = {top_left_x}')
        
        top_left_y = box_info[1 * int(box_num) + n] * scaley
        #print(f'top_left_y = box_info[{1 * int(box_num) + n}] * {scaley} = {top_left_y}')
        
        bottom_right_x = box_info[2 * int(box_num) + n] * scalex
        #print(f'bottom_left_x = box_info[{2 * int(box_num) + n}] * {scalex} = {bottom_right_x}')
        
        bottom_right_y = box_info[3 * int(box_num) + n] * scaley
        #print(f'bottom_left_y = box_info[{3 * int(box_num) + n}] * {scaley} = {bottom_right_y}')
        ##############################################################################
        print(" % s: class % d, box % d % d % d % d, score % f" % (
            label, ids, top_left_x, top_left_y, 
            bottom_right_x, bottom_right_y, score))
        cv2.rectangle(bgr_img, (int(top_left_x), int(top_left_y)), 
                (int(bottom_right_x), int(bottom_right_y)), colors[n % 6])
        p3 = (max(int(top_left_x), 15), max(int(top_left_y), 15))
        cv2.putText(bgr_img, label, p3, cv2.FONT_ITALIC, 0.6, colors[n % 6], 1)

    output_file = os.path.join("./out", "out_" + os.path.basename(image_file))
    print("output:%s" % output_file)
    cv2.imwrite(output_file, bgr_img)
    print("success!")
    #plt.figure(figsize=(20,10))
    return bgr_img