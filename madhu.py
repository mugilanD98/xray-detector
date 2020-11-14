import tensorflow as tf
import numpy as np
import streamlit as st
import keras
from PIL import Image
import requests
from io import BytesIO
st.set_option('deprecation.showfileUploaderEncoding',False)
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

path = st.text_input('enter image url ','https://storage.googleapis.com/kagglesdsdata/datasets/17810/23812/chest_xray/test/NORMAL/IM-0001-0001.jpeg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20201110%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20201110T221420Z&X-Goog-Expires=172799&X-Goog-SignedHeaders=host&X-Goog-Signature=6e7109157b51c5e4ffcd090b0f731ddd86ed306afc19d9c769495cc515b390ece403b652c83d42a99f31194f1ba84afb6b3c55ad07f4839022617d7ce32d2489db6cd6f004bba6884ea1a801601daefc525916c5f66b7479a26e0b1ec2578eb4575559a61b49adbb36f5668e95e5375f3c5d83fbb4fdf47b9c11db7db2070c655f3f67646ef4c88371255797939a87bfd424e1329ff3620d70d3ce0e163353d41bb184df5e70cadc3c826256a1cd75ab6df3684f1ff1f7e01c7b608d96c37517d535b00a6f4fc28d4932c5357ac69c071f2d696867f8554d981f1e692bbe2e675b28efc91f850db7dbfa43287eee81053172fa5c9261eeb5a3dc81a65f441a03')
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
