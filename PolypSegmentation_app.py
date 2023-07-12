import streamlit as st
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
import numpy as np
from PIL import Image
from train import iou

def read_image(image):
    newsize = (256,256)
    image = image.resize(newsize)
    st.image(image)
    return image

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
    image = Image.open(file).convert('RGB')
    image = read_image(image)
    h, w, _ = image.shape
    pre_image = model.predict(np.expand_dims(x, axis=0))[0] > 0.5
    white_line = np.ones((h, 10, 3)) * 255.0
    all_images = [
        image * 255.0, white_line,
        mask_parse(pre_image) * 255.0
    ]
    final_image = np.concatenate(all_images, axis=1)
    st.write("## Prediction Mask")
    st.image(final_image)