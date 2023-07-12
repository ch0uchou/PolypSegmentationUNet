import streamlit as st
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
import cv2
import numpy as np
from PIL import Image

def mask_parse(mask):
    mask = np.squeeze(mask)
    mask = [mask, mask, mask]
    mask = np.transpose(mask, (1, 2, 0))
    return mask

st.title('Polyp Segmentation')

st.header('Please upload an image')

file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

with CustomObjectScope({'iou': iou}):
    model = tf.keras.models.load_model("model.h5")

if file is not None:
    image = cv2.imread(file, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (256, 256))
    image = x/255.0

    pre_image = model.predict(np.expand_dims(image, axis=0))[0] > 0.5
    
    w, h = image.shape
    white_line = np.ones((h, 10, 3)) * 255.0
    
    all_images = [
        image, white_line,
        mask_parse(pre_image) 
    ]

    final_image = np.concatenate(all_images, axis=1)
    st.write("## Prediction Mask")
    st.image(final_image)