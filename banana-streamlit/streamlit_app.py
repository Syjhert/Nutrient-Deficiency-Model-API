import streamlit as st
import os
import numpy as np
from PIL import Image
from ultralytics import YOLO

@st.cache_resource
def load_model():
    return YOLO("banana-streamlit/yolo-banana-nutrient-best.pt")
    # model = YOLO("yolo_banana_nutrient_best.pt")


model = load_model()

classes = ['Boron', 'Calcium', 'Healthy', 'Iron', 'Magnesium', 'Manganese', 'Potassium', 'Sulfur', 'Zinc']

st.title("Banana Halation Basic Streamlit App")

uploaded_file = st.file_uploader("Upload a banana leaf image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # convert to RGB because some images have shape of (224, 224, 4) after preprocess,
    # so we convert them to (224, 224, 3)
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_container_width=True)

    if st.button("Predict"):
        results = model(image)
        pred = results[0].probs.top1
        confidence = results[0].probs.data[pred].item()

        st.header(f"Predicted Class: {classes[pred]} \nConfidence: {confidence*100:.2f}%")