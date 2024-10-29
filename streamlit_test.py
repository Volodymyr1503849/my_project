import numpy as np
import tensorflow as tf
from PIL import Image
import streamlit as st
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

model_cnn = load_model("model_keras.h5")

labels = [
    "Black-grass",
    "Charlock",
    "Cleavers",
    "Common Chickweed",
    "Common wheat",
    "Fat Hen",
    "Loose Silky-bent",
    "Maize",
    "Scentless Mayweed",
    "Shepherds Purse",
    "Small-flowered Cranesbill",
    "Sugar beet",
]

def preprocess_image(image):
    img = image.convert("RGB")  
    img = img.resize((150, 150)) 
    img = np.array(img).astype("float32") / 255 
    img = np.expand_dims(img, axis=0)  
    return img

st.title("Класифікація зображень")
uploaded_file = st.file_uploader("Завантажте зображення", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Завантажене зображення", use_column_width=True)
    img = preprocess_image(image)
    
    predictions = model_cnn.predict(img)
    predicted_class = labels[np.argmax(predictions)]
    
    st.write(f"Передбачений клас: {predicted_class}")
