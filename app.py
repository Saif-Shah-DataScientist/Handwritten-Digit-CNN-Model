import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import onnxruntime as ort
import pandas as pd

st.set_page_config(page_title="Digit Recognition", page_icon="✏️", layout="centered")

# Load ONNX model
@st.cache_resource
def load_model():
    return ort.InferenceSession("digit_model.onnx")

session = load_model()

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

st.title("✏️ Handwritten Digit Recognition (ONNX)")
st.subheader("Upload an image of a handwritten digit")

uploaded_file = st.file_uploader("Upload digit image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img = image.convert("L")
    img = img.resize((28, 28))
    img = ImageOps.invert(img)

    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    # Run inference
    result = session.run([output_name], {input_name: img_array})[0]

    predicted_digit = int(np.argmax(result))
    probabilities = result.flatten()

    # Display Prediction
    st.markdown(f"""
        <div style="
            padding:20px;
            background:#f0f0f0;
            border-radius:15px;
            text-align:center;
            font-size:26px;
            font-weight:bold;">
            🧠 Predicted Digit: <span style="color:#0078ff">{predicted_digit}</span>
        </div>
    """, unsafe_allow_html=True)

    # Show probability chart
    st.write("### 🔢 Prediction Probabilities")
    df = pd.DataFrame({"Digit": list(range(10)), "Probability": probabilities})
    st.bar_chart(df, x="Digit", y="Probability")

else:
    st.info("📤 Upload a digit image to begin.")
