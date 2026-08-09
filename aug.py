import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image


# Page configuration
st.set_page_config(
    page_title="DeepFake Detection System",
    layout="centered"
)


# Title
st.title("DeepFake Image Detection System")

st.write("Deep Learning Based Real and Fake Face Detection")

st.write(
    "Upload a face image to check whether it is Real or Fake."
)

st.divider()


# Load trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        "deep_model.keras",
        compile=False
    )
    return model


# Load model
try:
    model = load_model()
    st.success("Model loaded successfully.")

except Exception as e:
    st.error("Model could not be loaded.")
    st.write("Error:", e)
    st.stop()


# Upload image
st.subheader("Upload Image")

uploaded_file = st.file_uploader(
    "Select a face image",
    type=["jpg", "jpeg", "png"]
)


# Image prediction
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Selected Image")

    st.image(
        image,
        caption="Uploaded Face Image",
        width=400
    )

    st.divider()

    if st.button("Detect Image"):

        # Convert image to NumPy array
        img = np.array(image)

        # Resize to 64 x 64
        img = cv2.resize(
            img,
            (64, 64)
        )

        # Normalize
        img = img.astype("float32") / 255.0

        # Add batch dimension
        img = np.expand_dims(
            img,
            axis=0
        )

        # Prediction
        prediction = model.predict(
            img,
            verbose=0
        )

        # Get predicted class
        predicted_class = np.argmax(
            prediction[0]
        )

        # Get confidence
        confidence = (
            float(prediction[0][predicted_class]) * 100
        )

        # Result
        st.subheader("Result")

        if predicted_class == 0:

            st.error("The image is FAKE")
            result = "FAKE"

        else:

            st.success("The image is REAL")
            result = "REAL"

        st.write("Prediction:", result)

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )


# About project
st.divider()

st.subheader("About the Project")

st.write(
    "This project uses a Convolutional Neural Network (CNN) "
    "to detect whether a face image is Real or Fake."
)

st.write(
    "The uploaded image is resized to 64 × 64 pixels "
    "before prediction."
)

st.write(
    "The system provides the predicted class and "
    "confidence percentage."
)
