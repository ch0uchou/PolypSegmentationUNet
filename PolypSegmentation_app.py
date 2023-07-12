import streamlit as st
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
import cv2
import numpy as np
from PIL import Image
from train import iou

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
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)

    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (256, 256))
    image = image/255.0

    st.image(image,channels="BGR")

    pre_image = model.predict(np.expand_dims(image, axis=0))[0] > 0.5
    
    final_pre = np(mask_parse(pre_image))

    final_image = cv2.imwrite("final.png",final_pre)
    st.image(final_image)
    
    st.write("## Prediction Mask")
    # st.image(final_image)