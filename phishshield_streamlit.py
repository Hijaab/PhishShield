import os
os.environ["STREAMLIT_DISABLE_FILE_WATCHER"] = "true"

import streamlit as st
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import numpy as np
from PIL import Image
import pandas as pd

# ----- Setup -----
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# ----- Define Model -----
def GlobalAvgPooling(x):
    return x.mean(dim=2).mean(dim=2)

class ENSModel(nn.Module):
    def __init__(self):
        super(ENSModel, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.avgpool = GlobalAvgPooling
        self.efn = EfficientNet.from_pretrained('efficientnet-b0')
        self.dense_output = nn.Linear(1280, 1)

    def forward(self, x):
        feat = self.efn.extract_features(x)
        pooled = self.avgpool(feat)
        output = self.dense_output(pooled)
        return self.sigmoid(output)

# ----- Load Models -----
@st.cache_resource
def load_models():
    models = []
    for i in range(1, 5):
        model = ENSModel().to(device)
        model_path = os.path.join("model", f"efficientnet_model_{i}.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        models.append(model)
    return models

models = load_models()

# ----- Preprocess -----
def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file).convert('RGB')
    image = image.resize((512, 512))
    img = np.array(image) / 255.0
    img = np.transpose(img, (2, 0, 1)).astype(np.float32)
    img_tensor = torch.tensor(img).unsqueeze(0).to(device)
    return img_tensor, image

# ----- Streamlit UI -----
st.set_page_config(page_title="PhishShield", layout="wide")

st.markdown("""
# 🛡️ PhishShield: Steganography Detector  
Upload an image and our AI ensemble will analyze it for hidden steganographic content.
""")

uploaded_file = st.file_uploader("Upload an Image", type=list(ALLOWED_EXTENSIONS), label_visibility="collapsed")

left_col, right_col = st.columns([1, 1.5])

if uploaded_file:
    with st.spinner("🔍 Analyzing..."):
        img_tensor, display_image = preprocess_image(uploaded_file)
        predictions = []
        scores = {}
        with torch.no_grad():
            for idx, model in enumerate(models):
                output = model(img_tensor)
                score = output.item()
                predictions.append(score)
                scores[f"Model {idx+1}"] = round(score * 100, 2)

        avg_score = round(sum(predictions) / len(predictions) * 100, 2)
        result = 'Stego' if avg_score >= 60 else 'Non-Steg'
        result_color = 'red' if result == 'Stego' else 'green'

        # ----- LEFT COLUMN -----
        with left_col:
            st.subheader("📷 Uploaded Image")
            st.image(display_image, use_container_width=True)
            st.markdown("---")
            st.markdown("### ℹ️ Model Output (Raw)")
            with st.expander("See JSON Output"):
                st.json(scores)

        # ----- RIGHT COLUMN -----
        with right_col:
            st.subheader("🧠 AI Prediction")
            st.markdown(f"<h3 style='color:{result_color}'>Prediction: {result}</h3>", unsafe_allow_html=True)
            st.progress(int(avg_score))

            col1, col2 = st.columns(2)
            col1.metric("Confidence", f"{avg_score:.2f}%", delta=None)
            col2.metric("Models Used", f"{len(models)}")

            st.subheader("📊 Model Agreement Chart")
            score_df = pd.DataFrame(scores.items(), columns=["Model", "Score (%)"])
            st.bar_chart(score_df.set_index("Model"))

            st.subheader("📈 Trend Simulation")
            # Simulate fake trend just for busy-looking UI
            fake_trend = pd.DataFrame({
                "Ensemble Trend": np.convolve(predictions, np.ones(2)/2, mode='same')
            })
            st.line_chart(fake_trend)
