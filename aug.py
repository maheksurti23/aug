import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="DeepFake Detection System",
    page_icon="search",
    layout="centered"
)

# ============================================================
# TITLE
# ============================================================

st.title("🔍 DeepFake Image Detection System")

st.write(
    "Deep Learning Based Real and Fake Face Detection"
)

st.write(
    "Upload a face image to check whether it is Real or Fake."
)

st.divider()

# ============================================================
# LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        "deepfake_model.keras"
    )

    return model


try:

    model = load_model()

    st.success("Model loaded successfully")

except Exception:

    st.error("Model could not be loaded.")

    st.warning(
        "Please keep 'deepfake_model.keras' "
        "in the same folder as this Python file."
    )

    st.stop()


# ============================================================
# UPLOAD IMAGE
# ============================================================

st.subheader("Upload Image")

uploaded_file = st.file_uploader(
    "Select a face image",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# IMAGE PREDICTION
# ============================================================

if uploaded_file is not None:

    # Read image
    image = Image.open(uploaded_file)

    # Convert to RGB
    image = image.convert("RGB")

    # Display image
    st.subheader("Selected Image")

    st.image(
        image,
        caption="Uploaded Face Image",
        width=400
    )

    st.divider()

    # Detect button
    if st.button("Detect Image"):

        # Convert image to NumPy array
        img = np.array(image)

        # Resize according to your project
        # 64 × 64
        img = cv2.resize(
            img,
            (64, 64)
        )

        # Normalize image
        img = img / 255.0

        # Add batch dimension
        img = np.expand_dims(
            img,
            axis=0
        )

        # ====================================================
        # PREDICTION
        # ====================================================

        prediction = model.predict(img)

        probability = float(
            prediction[0][0]
        )

        # ====================================================
        # RESULT
        # ====================================================

        if probability >= 0.5:

            result = "FAKE"
            confidence = probability * 100

        else:

            result = "REAL"
            confidence = (1 - probability) * 100


        # ====================================================
        # DISPLAY RESULT
        # ====================================================

        st.subheader("Result")

        if result == "FAKE":

            st.error(
                "The image is FAKE"
            )

        else:

            st.success(
                "The image is REAL"
            )

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )


# ============================================================
# ABOUT PROJECT
# ============================================================

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