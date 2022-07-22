import cv2
import time
from acl_model import Model

device_id = 0
model_path = "model/yolov5s_aipp.om"
model = Model(device_id, model_path, model_type="yolov5")
img_path = "data/kite.jpg"
img_org_bgr = cv2.imread(img_path)

for _ in range(1000):
    bboxes = model.run1(img_org_bgr)