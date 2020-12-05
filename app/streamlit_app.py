import os
import time

import altair as alt
import keras
import numpy as np
import pandas as pd
import streamlit as st
from keras.utils import plot_model
from PIL import Image
from streamlit_drawable_canvas import st_canvas

#import footer

model_list = os.listdir("mlmodels")

model_path = st.sidebar.selectbox(
    "Select ML model:", tuple(model_list)
)

model = keras.models.load_model(os.path.join("mlmodels", model_path))

st.title("MNIST DNN runner")


# Specify canvas parameters in application
stroke_width = 35
stroke_color = "#000"
bg_color = "#eee" 
drawing_mode = "freedraw" 

col21, col22 = st.beta_columns(2)

with col22:
    st.text(".. then click the button ")
    update_button = st.button('Classify Digit')

with col21:
    st.text("draw in the canvas ..")
    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color= bg_color,
        background_image=None,
        update_streamlit=update_button,
        height=280,
        width=280,
        drawing_mode=drawing_mode,
        key="canvas",
    )

# Do something interesting with the image data
if (canvas_result.image_data is not None):
    scaled_img = canvas_result.image_data[::10,::10,:]

    dnn_img = scaled_img[:,:,0]
    dnn_img = dnn_img/255.0
    dnn_img += 1.0
    dnn_img[dnn_img>1.0] = 0.0
    result = model.predict(dnn_img.reshape([1, 28,28,1]))[0]

    chart = alt.Chart(pd.DataFrame({
            'numbers': list(range(10)),
            'probability': result
        })).mark_bar(size=30).encode(
        x=alt.X('numbers',
        scale=alt.Scale(domain=(-0.5, 9.5), clamp=True), axis=alt.Axis(tickMinStep=1)),
        y='probability',
    )
    col11, col12 = st.beta_columns(2)
    col11.markdown("Image passed to DNN ( 28px by 28px ): ")
    col12.image(scaled_img)
    predict_nr = np.argmax(result)
    col21, col22 = st.beta_columns(2)
    col11.markdown(f'''Digit Classification:   <font size="6"> {predict_nr} </font>''', unsafe_allow_html=True)

    st.altair_chart(chart, use_container_width=True)
    
#footer.footer() todo - bug found "cannot import name ‘div’" on heroku