import streamlit as st
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
import numpy as np
from PIL import Image
from train import iou

def mask_parse(mask):
    mask = np.squeeze(mask)
    mask = [mask, mask, mask]
    mask = np.transpose(mask, (1, 2, 0))
    return mask

def run(image, model):
    newsize = (256,256)
    image = image.resize(newsize)
    pre_mask = model.predict(np.expand_dims(image, axis=0))[0] > 0.5
    return pre_mask

st.title('Polyp Segmentation')

st.header('Please upload an image')

file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

with CustomObjectScope({'iou': iou}):
    model = tf.keras.models.load_model("model.h5")

if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    pre_mask = run(image, model)
    pre_mask = mask_parse(pre_mask) * 255.0

    st.write("## Prediction Mask")
    st.image(pre_mask, clamp=True, channels='BGR')