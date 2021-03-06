from numpy.lib import npyio
import streamlit as st
import os
import numpy as np
from PIL import Image
from pathlib import Path

st.title("Table Detector")
import TableDetector.deskew as deskew
import TableDetector.table_detection as table_detector

# import matrix libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt

# import some common detectron2 utilities
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data import DatasetCatalog, MetadataCatalog

setup_logger()

#create detectron config
cfg = get_cfg()
#set yaml
cfg.merge_from_file('content/All_X152.yaml')
#set model weights
cfg.MODEL.DEVICE='cpu'
cfg.MODEL.WEIGHTS = 'content/model_final.pth' # Set path model .pth
predictor = DefaultPredictor(cfg)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    document_img = np.array(image)
    if len(document_img.shape) > 1:
        document_img  = cv2.cvtColor(document_img , cv2.COLOR_RGB2BGR)
    deskewed_image = deskew.deskewImage(document_img)
    document_img = table_detector.get_prediction(deskewed_image, predictor)
    st.image(document_img, caption='Result Image.', use_column_width=True)
