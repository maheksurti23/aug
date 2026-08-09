import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="DeepFake Detection System",
    page_icon="🔍",
    layout="centered"
)

# Title
st.title("DeepFake Image Detection System")

st.write("Deep Learning Based Real and Fake Face Detection")
st.write("Upload a face image to check whether it is Real or Fake.")

st.divider()


# Load trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        "deep_model.keras",
        compile=False
    )
    return model


try:
    model = load_model()
    st.success("Model loaded successfully")

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


# Prediction
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Selected Image")

    st.image(
        image,
        caption="Uploaded Face Image",
        width=400
    )

    if st.button("Detect Image"):

        # Convert RGB image to NumPy
        img = np.array(image)

        # IMPORTANT:
        # Your original training code uses cv2.imread(),
        # which reads images in BGR format.
        img = cv2.cvtColor(
            img,
            cv2.COLOR_RGB2BGR
        )

        # Resize to the same size used during training
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

        # Model prediction
        prediction = model.predict(
            img,
            verbose=0
        )

        # Get predicted class
        predicted_class = np.argmax(
            prediction[0]
        )

        # Get confidence
        confidence = float(
            prediction[0][predicted_class]
        ) * 100

        # Your training code creates two classes.
        # Assuming:
        # 0 = Fake
        # 1 = Real

        if predicted_class == 0:
            result = "FAKE"
            st.error("The image is FAKE")
        else:
            result = "REAL"
            st.success("The image is REAL")

        st.write(
            f"Prediction: {result}"
        )

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )
