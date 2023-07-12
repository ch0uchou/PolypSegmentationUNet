import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)

def run(image, model):
    pre_mask = model.predict(np.expand_dims(image, axis=0))[0] > 0.5
    pre_mask = np.squeeze(pre_mask)
    pre_mask = [pre_mask, pre_mask, pre_mask]
    pre_mask = np.transpose(pre_mask, (1, 2, 0))
    return pre_mask * 255

st.title('Polyp Segmentation')

st.header('Please upload an image')

file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

model = tf.keras.models.load_model("model.h5")
# model.evaluaate()

if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    # pre_mask = run(image, model)

    st.write("## Prediction Mask")
    # st.image(pre_mask)