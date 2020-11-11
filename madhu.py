import tensorflow as tf
import numpy as np
import streamlit as st
import keras
from PIL import Image
import requests
from io import BytesIO
st.set_option('deprecation.showfileUploaderEncoding',False)
st.title("xray")
st.text("provide URL")
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

path = st.text_input('enter image url ','https://images.app.goo.gl/y8Nd9zX2u6pSMCW96')
if path is not None:
    content=requests.get(path).content

    st.write("predicted class:")
    with st.spinner('classify'):
        lable=classes[int(model.predict(decode_img(content)))]
        st.write(lable)
    st.write("")
    image=Image.open(BytesIO(content))
    st.image(image,caption='classify xray',use_column_width=True)
