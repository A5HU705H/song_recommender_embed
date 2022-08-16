import streamlit as st
from PIL import Image
import cv2
import numpy as np
import cv2
import tensorflow as tf
import base64

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('bg-5.jpg')

st.title("Welcome to Song Recommendation by Emotion Detection")
picture=st.camera_input('Capture Face for Recognition')
if picture is not None:
    st.image(picture)
    img=Image.open(picture)
    picture=np.array(img)
    cv2.imwrite('photo.jpg',picture)
    ret,jpeg_1=cv2.imencode('.jpg',picture)
    model=tf.keras.models.load_model('my_model.h5')
    jpeg_1=np.array(jpeg_1)
    cv2.imwrite('photo1.jpg',jpeg_1)
    jpeg_1=cv2.resize(jpeg_1,(48,48))
    jpeg_1=np.expand_dims(jpeg_1, axis=0)
    result=model.predict(jpeg_1)
    max=0
    maxpos=0
    for i in range(7):
        if(result[0][i]>max):
            max=result[0][i]
            maxpos=i
    anger_embed="""
    <iframe style="border-radius:12px" src="https://open.spotify.com/embed/playlist/71Xpaq3Hbpxz6w9yDmIsaH?utm_source=generator" width="100%" height="380" frameBorder="0" allowfullscreen="" allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture"></iframe>"""
    disgust_embed="""
    <iframe style="border-radius:12px" src="https://open.spotify.com/embed/playlist/3qgzMg4m5tvf16PzlPgGa9?utm_source=generator" width="100%" height="380" frameBorder="0" allowfullscreen="" allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture"></iframe>"""
    fear_embed="""
    <iframe style="border-radius:12px" src="https://open.spotify.com/embed/playlist/7rzS9iLiqjy65AsZd9qinf?utm_source=generator" width="100%" height="380" frameBorder="0" allowfullscreen="" allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture"></iframe>"""
    happy_embed="""
    <iframe style="border-radius:12px" src="https://open.spotify.com/embed/playlist/37i9dQZF1DXdPec7aLTmlC?utm_source=generator&theme=0" width="100%" height="380" frameBorder="0" allowfullscreen="" allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture"></iframe>"""
    neutral_embed="""
    <iframe style="border-radius:12px" src="https://open.spotify.com/embed/playlist/37i9dQZF1DWTC99MCpbjP8?utm_source=generator" width="100%" height="380" frameBorder="0" allowfullscreen="" allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture"></iframe>"""
    sad_embed="""
    <iframe style="border-radius:12px" src="https://open.spotify.com/embed/playlist/37i9dQZF1DX7qK8ma5wgG1?utm_source=generator" width="100%" height="380" frameBorder="0" allowfullscreen="" allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture"></iframe>"""
    surprise_embed="""
    <iframe style="border-radius:12px" src="https://open.spotify.com/embed/playlist/0X0ZZTJ6z2yxX5Uu7R7j3G?utm_source=generator" width="100%" height="380" frameBorder="0" allowfullscreen="" allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture"></iframe>"""
    dict={0:'anger',1:'disgust',2:'fear',3:'happiness',4:'neutral',5:'sadness',6:'surprise'}
    prediction=dict.get(maxpos)
    st.write("Emotion detected is: ",prediction)
    dict1={0:anger_embed,
           1:disgust_embed,
           2:fear_embed,
           3:happy_embed,
           4:neutral_embed,
           5:sad_embed,
           6:surprise_embed
          }
    song_link=dict1.get(maxpos)
    st.markdown(song_link,unsafe_allow_html=True)