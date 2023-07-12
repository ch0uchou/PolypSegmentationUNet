import streamlit as st
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
import numpy as np
from PIL import Image
from train import iou

def run(image, model):
    image = image.resize(256, 256)
    image = image/255.0
    pre_mask = model.predict(np.expand_dims(image, axis=0))[0] > 0.5
    pre_mask = np.squeeze(pre_mask)
    pre_mask = [pre_mask, pre_mask, pre_mask]
    pre_mask = np.transpose(pre_mask, (1, 2, 0))
    return pre_mask * 255

st.title('Polyp Segmentation')

st.header('Please upload an image')

file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

with CustomObjectScope({'iou': iou}):
    model = tf.keras.models.load_model("model.h5")
# model.evaluaate()

if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    pre_mask = run(image, model)

    st.write("## Prediction Mask")
    st.image(pre_mask)