import tensorflow as tf
import numpy as np
import streamlit as st
import keras
from PIL import Image
import requests
from io import BytesIO
st.set_option('deprecation.showfileUploaderEncoding',False)
st.markdown("<h1 style='text-align: right; color: black;'>mugilan D (19CSEG025)<h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: black;'>Radiology Assistant<h1>", unsafe_allow_html=True)
st.text("provide URL of x-ray to classify")
st.text("image should be any one of given formats (jpeg,png,jpg)")
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('pneumoniamugi.h5')
    return model

with st.spinner('loading model into memory....'):
    model = load_model()

classes=['pneumonia','normal']

def scale(image):
    image=tf.cast(image,tf.float32)
    image /=255.0
    return tf.image.resize(image,[200,200])

def decode_img(image):
    img=tf.image.decode_jpeg(image,channels=1)
    img=scale(img) 
    return np.expand_dims(img,axis=0)

path = st.text_input('enter image url ','https://prod-images-static.radiopaedia.org/images/52302679/2359ab3b9b9b0bfc9874f158bda46a_jumbo.jpeg')
if path is not None:
    content=requests.get(path).content

    st.write("predicted class:")
    with st.spinner('classify'):
        lab=model.predict(decode_img(content))
        if lab >= 0.56:
           lable=classes[1]
        else:
           lable=classes[0]
        
        st.write(lable)
    st.write("")
    image=Image.open(BytesIO(content))
    st.image(image,caption='classify xray',use_column_width=True)
