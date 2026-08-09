import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

# Page
st.set_page_config(
    page_title="DeepFake Detection",
    layout="centered"
)

# Title
st.title("DeepFake Image Detection System")
st.write("Upload an image to check whether it is Real or Fake.")

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("deep_model.keras")

model = load_model()

# Upload image
file = st.file_uploader(
    "Upload Face Image",
    type=["jpg", "jpeg", "png"]
)

# Prediction
if file is not None:

    image = Image.open(file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        width=400
    )

    if st.button("Detect Image"):

        # Preprocessing
        img = np.array(image)
        img = cv2.resize(img, (64, 64))
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)

        # Prediction
        prediction = model.predict(img, verbose=0)
        probability = float(prediction[0][0])

        # Result
        if probability >= 0.5:
            st.error("The image is FAKE")
            confidence = probability * 100
        else:
            st.success("The image is REAL")
            confidence = (1 - probability) * 100

        st.write(f"Confidence: {confidence:.2f}%")
