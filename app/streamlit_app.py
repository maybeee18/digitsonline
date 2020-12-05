import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import altair as alt
import time

import keras

import streamlit.components.v1 as components

st.title('MNIST DNN explorer')

model = keras.models.load_model("../mymodel/")


# Specify canvas parameters in application
stroke_width = 35
stroke_color = "#000"
bg_color = "#eee"
drawing_mode = "freedraw"


# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    background_image= None,
    update_streamlit=True,
    height=280,
    width=280,
    drawing_mode=drawing_mode,
    key="mycanvas",
)

# Do something interesting with the image data and paths

    
if canvas_result.image_data is not None and st.button("Predict"):
    progress = st.progress(1)

    scaled_img = canvas_result.image_data[::10,::10,:]
    progress.progress(20)
    st.text("Image passed to DNN (28x28): ")

    st.image(scaled_img)
    progress.progress(50)

    dnn_img = scaled_img[:,:,0]
    dnn_img = dnn_img/255.0
    dnn_img += 1.0
    dnn_img[dnn_img>1.0] = 0.0
    result = model.predict(dnn_img.reshape([1, 28,28,1]))[0]
    progress.progress(70)

    chart = alt.Chart(pd.DataFrame({
            'numbers': list(range(10)),
            'probability': result
        })).mark_bar().encode(
        x='numbers',
        y='probability',
    )
    progress.progress(100)
    st.altair_chart(chart, use_container_width=True)
    st.text(np.argmax(result))