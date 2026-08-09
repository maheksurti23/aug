import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import os

# Page configuration
st.set_page_config(
    page_title="DeepFake Detection System",
    layout="centered"
)

# Title
st.title("DeepFake Image Detection System")

st.write(
    "Deep Learning Based Real and Fake Face Detection"
)

st.write(
    "Upload a face image or select a sample image to check the result."
)

st.divider()

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "deep_model.keras",
        compile=False
    )

try:
    model = load_model()
    st.success("Model loaded successfully")

except Exception as e:
    st.error("Model could not be loaded.")
    st.write("Error:", e)
    st.stop()


# Image selection
st.subheader("Select Image")

option = st.radio(
    "Choose an option:",
    ["Upload Image", "Use Sample Image"]
)

# UPLOAD IMAGE

if option == "Upload Image":

    uploaded_file = st.file_uploader(
        "Select a face image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

# SAMPLE IMAGE

else:

    sample_images = {
        "Sample Image 1": "samples/easy_104_1000.jpg",
        "Sample Image 2": "samples/real_00004.jpg"
    }

    selected_image = st.selectbox(
        "Select a sample image:",
        list(sample_images.keys())
    )

    image_path = sample_images[selected_image]

    if os.path.exists(image_path):

        image = Image.open(image_path).convert("RGB")

    else:

        st.error(
            "Sample image not found. "
            "Please upload the images to the samples folder."
        )

        st.stop()

# DISPLAY IMAGE

if "image" in locals():

    st.subheader("Selected Image")

    st.image(
        image,
        caption="Input Face Image",
        width=400
    )

    st.divider()

    # DETECT IMAGE

    if st.button("Detect Image"):

        # Convert image to NumPy
        img = np.array(image)

        # Convert RGB to BGR
        img = cv2.cvtColor(
            img,
            cv2.COLOR_RGB2BGR
        )

        # Resize to 64 × 64
        # Same size as your training code
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

        # Confidence
        confidence = (
            float(prediction[0][predicted_class])
            * 100
        )

        # RESULT

        st.subheader("Detection Result")

        # Assuming:
        # 0 = Fake
        # 1 = Real

        if predicted_class == 0:

            st.error("The image is FAKE")

            result = "FAKE"

        else:

            st.success("The image is REAL")

            result = "REAL"


        st.write(
            "Prediction:",
            result
        )

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

# ABOUT PROJECT

st.divider()

st.subheader("About the Project")

st.write(
    "This project uses a Convolutional Neural Network (CNN) "
    "to detect whether a face image is Real or Fake."
)

st.write(
    "The model was trained using the Real and Fake Face "
    "Detection dataset."
)

st.write(
    "Input images are resized to 64 × 64 pixels before prediction."
)
