import cv2
import numpy as np

def get_prediction(img, predictor):
    outputs = predictor(img)
    color = (255, 0, 0) 
    thickness = 2
    for x1, y1, x2, y2 in outputs["instances"].get_fields()["pred_boxes"].tensor.to("cpu").numpy():
        start_point = (x1, y1) 
        end_point = (x2, y2) 
        img = cv2.rectangle(np.array(img, copy=True), start_point, end_point, color, thickness) 
    return img