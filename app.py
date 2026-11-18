import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from transformers import pipeline
import torch 

XRAY_LABEL = "a medical chest x-ray radiograph"
NON_XRAY_LABEL = (
    "a regular photo of an everyday object, animal, person, or scenery"
)
XRAY_CANDIDATE_LABELS = [XRAY_LABEL, NON_XRAY_LABEL]

# Load model
model = tf.keras.models.load_model("pneumonia_model.h5")

# Page config
st.set_page_config(
    page_title="AI Pneumonia Detector",
    page_icon="🫁",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
body {
    background-color: #0f2027;
}
.main {
    background: linear-gradient(to right, #2193b0, #6dd5ed);
    padding: 30px;
    border-radius: 20px;
}
h1, h2, h3 {
    text-align: center;
    color: #ffffff;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main'>", unsafe_allow_html=True)

st.title("🫁 AI Pneumonia Detection System")
st.write("### Upload a Chest X-ray Image")
st.write("This AI system helps detect pneumonia using deep learning.")

uploaded_file = st.file_uploader(
    "Choose an X-ray image",
    type=["jpg", "jpeg", "png"]
)


@st.cache_resource
def load_xray_validator():
    return pipeline(
        "zero-shot-image-classification",
        model="openai/clip-vit-base-patch32",
        framework="pt",
    )
def is_chest_xray(image):
    validator = load_xray_validator()
    scores = {
        result["label"]: result["score"]
        for result in validator(image, candidate_labels=XRAY_CANDIDATE_LABELS)
    }
    return scores[XRAY_LABEL] >= scores[NON_XRAY_LABEL]

def predict(image):
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    probability = float(prediction[0][0])
    confidence = max(probability, 1.0 - probability)
    return probability, confidence

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_container_width=True)

    if st.button("🔍 Analyze X-ray"):
        with st.spinner("AI is analyzing the image..."):
            if not is_chest_xray(image):
                st.error(
                    "Invalid Image: This does not appear to be a chest X-ray. "
                    "Please upload a valid chest X-ray image."
                )
            else:
                probability, confidence = predict(image)
                if probability > 0.5:
                    st.error(
                        f"🚨 Pneumonia Detected (Confidence: {confidence:.2%})"
                    )
                    st.write("Please consult a medical professional.")
                else:
                    st.success(
                        f"✅ Normal Chest X-ray (Confidence: {confidence:.2%})"
                    )
                    st.write("No signs of pneumonia detected.")

st.markdown("</div>", unsafe_allow_html=True)

# streamlit run app.py
