import streamlit as st
import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

# streamlit cloud cant find the model, so we debug :)
print("Current directory:", os.getcwd()) 

model = load_model('streamlit/rice_deficiency_model.h5')
# model = load_model('rice_deficiency_model.h5')

classes = ['Nitrogen(N)', 'Phosphorus(P)', 'Potassium(K)']

def preprocess_image(image_data):
    img = image_data.resize((224, 224))
    img_tensor = keras_image.img_to_array(img)
    print(img_tensor.shape)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.
    return img_tensor

st.title("Rice Halation Basic Streamlit App")

uploaded_file = st.file_uploader("Upload a rice leaf image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # convert to RGB because some images have shape of (224, 224, 4) after preprocess,
    # so we convert them to (224, 224, 3)
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_container_width=True)

    if st.button("Predict"):
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        class_idx = np.argmax(prediction[0])
        class_label = classes[class_idx]
        confidence = float(prediction[0][class_idx])

        st.header(f"Predicted Class: {class_label} \nConfidence: {confidence*100:.2f}%")