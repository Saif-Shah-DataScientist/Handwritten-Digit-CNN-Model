# app.py

import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import pandas as pd

# ----------------------------------------
# PAGE SETTINGS
# ----------------------------------------
st.set_page_config(
    page_title="Digit Recognition",
    page_icon="🔢",
    layout="centered"
)

st.title("🖌️ Handwritten Digit Recognition App")
st.write("Upload an image containing a digit (0–9) and the model will predict it.")


# ----------------------------------------
# LOAD MODEL
# ----------------------------------------
@st.cache_resource
def load_digit_model():
    return load_model("digit_model.h5")   # Make sure this file is in same folder

model = load_digit_model()


# ----------------------------------------
# FILE UPLOADER
# ----------------------------------------
uploaded_file = st.file_uploader(
    "Upload a digit image (PNG/JPG/JPEG):",
    type=["png", "jpg", "jpeg"]
)


# ----------------------------------------
# MAIN APP LOGIC
# ----------------------------------------
if uploaded_file is not None:

    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocessing for CNN
    image = image.convert("L")           # Convert to grayscale
    image = image.resize((28, 28))       # Resize to 28x28
    image = ImageOps.invert(image)       # Invert (white digit on black)
    img_arr = np.array(image) / 255.0    # Normalize
    img_arr = img_arr.reshape(1, 28, 28, 1)  # CNN expects this shape

    # Prediction
    prediction = model.predict(img_arr)
    predicted_digit = int(np.argmax(prediction))

    # ----------------------------------------
    # MODEL PREDICTION DISPLAY
    # ----------------------------------------
    st.markdown("### 🧠 Model Prediction")
    st.success(f"Predicted Digit: **{predicted_digit}**")

    # ----------------------------------------
    # BEAUTIFUL PREDICTION PROBABILITY TABLE
    # ----------------------------------------
    st.markdown("### 📊 Prediction Probabilities")

    prob_array = prediction.flatten()
    digits = list(range(10))

    df = pd.DataFrame([prob_array], columns=[str(d) for d in digits])

    # Styled table
    st.dataframe(
        df.style.background_gradient(cmap="Blues")
                .format("{:.4f}")   # 4 decimal places
    )
