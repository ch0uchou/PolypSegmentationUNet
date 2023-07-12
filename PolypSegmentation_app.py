import streamlit as st
from keras.models import load_model
import numpy as np
from PIL import Image


def run(image, model):
    pre_mask = model.predict(np.expand_dims(image, axis=0))[0] > 0.5
    return pre_mask

st.title('Polyp Segmentation')

st.header('Please upload an image')

file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

model = load_model('./model.h5')

if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    pre_mask = run(image, model)

    st.write("## Prediction Mask")
    st.image(pre_mask)