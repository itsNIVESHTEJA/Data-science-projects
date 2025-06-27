import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Constants
IMG_SIZE = (128, 128)
CLASS_NAMES = ['Apple Black Rot', 'Apple Cedar Rust', 'Apple Scab', 'Apple Healthy']

# Load model
model = load_model("plant_model_small.h5")


# Streamlit UI
st.title(" Apple Leaf Disease Classifier")
st.markdown("Upload a leaf image to detect the disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess
    image = image.resize(IMG_SIZE)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0]
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction)

    # Result
    st.markdown(f"### 🩺 Prediction: **{predicted_class}**")
    st.markdown(f"### 🔬 Confidence: **{confidence * 100:.2f}%**")
