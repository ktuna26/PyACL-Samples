from constant import ACL_ERROR_NONE
import cv2
import numpy as np

def check_ret(message, ret):
    if ret != ACL_ERROR_NONE:
        raise Exception("{} failed ret={}"
                        .format(message, ret))
    # else:
    #     print("{} success".format(message))
    
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