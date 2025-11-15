import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tflite_runtime.interpreter as tflite  # <-- IMPORTANT

st.set_page_config(page_title="Digit Recognition", page_icon="✏️", layout="centered")

# -------------------------------------------------------
# Load TFLite Model
# -------------------------------------------------------
@st.cache_resource
def load_tflite_model():
    interpreter = tflite.Interpreter(model_path="digit_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

st.title("✏️ Handwritten Digit Recognition (TFLite)")
st.subheader("Upload a digit image & let the model predict")

uploaded_file = st.file_uploader("Upload your digit image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = image.convert("L")
    img = img.resize((28, 28))
    img = ImageOps.invert(img)

    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    interpreter.set_tensor(input_details[0]["index"], img_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]["index"])

    predicted_digit = int(np.argmax(output_data))
    probabilities = output_data.flatten()

    st.markdown(f"""
        <div style="
            padding:20px;
            background:#f5f5f5;
            border-radius:15px;
            text-align:center;
            font-size:24px;
            font-weight:bold;
        ">
            🧠 Predicted Digit: <span style="color:#0a84ff">{predicted_digit}</span>
        </div>
    """, unsafe_allow_html=True)

    st.write("### 🔢 Prediction Probabilities")

    import pandas as pd
    df = pd.DataFrame({"Digit": list(range(10)), "Probability": probabilities})
    st.bar_chart(df, x="Digit", y="Probability")
else:
    st.info("📤 Please upload an image to get started.")
