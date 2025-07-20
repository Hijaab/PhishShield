import os
os.environ["STREAMLIT_DISABLE_FILE_WATCHER"] = "true"

import streamlit as st
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import numpy as np
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# ----- Model Class -----
def GlobalAvgPooling(x):
    return x.mean(dim=2).mean(dim=2)

class ENSModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.avgpool = GlobalAvgPooling
        self.efn = EfficientNet.from_pretrained("efficientnet-b0")
        self.dense_output = nn.Linear(1280, 1)

    def forward(self, x):
        feat = self.efn.extract_features(x)
        pooled = self.avgpool(feat)
        output = self.dense_output(pooled)
        return self.sigmoid(output)

@st.cache_resource
def load_models():
    models = []
    for i in range(1, 5):
        model = ENSModel().to(device)
        path = os.path.join("model", f"efficientnet_model_{i}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file missing: {path}")
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        models.append(model)
    return models

models = load_models()

def preprocess_image(file):
    img = Image.open(file).convert("RGB")
    img = img.resize((512, 512))
    arr = np.array(img) / 255.0
    arr = np.transpose(arr, (2, 0, 1)).astype(np.float32)
    return torch.tensor(arr).unsqueeze(0).to(device), img

def get_risk_label(score):
    if score >= 85:
        return "High Confidence"
    elif score >= 60:
        return "Moderate Confidence"
    else:
        return "Low Confidence"

# ----- UI -----
st.set_page_config("PhishShield", layout="wide")

st.markdown("""
    <style>
    html, body {
        background-color: #f5f7fa;
    }
    .main-title {
        text-align: center;
        font-size: 3em;
        font-weight: bold;
        color: #2d3436;
        padding-top: 1rem;
    }
    .sub-title {
        text-align: center;
        font-size: 1.1em;
        color: #636e72;
        margin-bottom: 2rem;
    }
    .card {
        background-color: white;
        padding: 1.5rem 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.06);
        margin-bottom: 2rem;
    }
    .footer {
        text-align: center;
        font-size: 0.9em;
        color: #95a5a6;
        margin-top: 3rem;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>PhishShield – Steganography Detection</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>AI-powered detection of hidden content in uploaded images</div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Upload an image (PNG, JPG, JPEG)", type=list(ALLOWED_EXTENSIONS))

if uploaded_file:
    img_tensor, img_disp = preprocess_image(uploaded_file)

    with st.spinner("Analyzing image..."):
        preds, scores_dict = [], {}
        with torch.no_grad():
            for i, model in enumerate(models):
                out = model(img_tensor).item()
                preds.append(out)
                scores_dict[f"Model {i+1}"] = round(out * 100, 2)

        avg_score = round(np.mean(preds) * 100, 2)
        variance = round(np.var(preds) * 10000, 2)
        label = "Stego" if avg_score >= 60 else "Non-Steg"
        interpretation = "Hidden content likely present." if label == "Stego" else "No hidden content detected."
        confidence = get_risk_label(avg_score)
        color = "#e74c3c" if label == "Stego" else "#2ecc71"

    col1, col2 = st.columns([1, 1.5], gap="large")

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Uploaded Image")
        st.image(img_disp, use_container_width=True)
        st.markdown(f"**Size:** {img_disp.size[0]} x {img_disp.size[1]}")
        st.markdown(f"**Mode:** {img_disp.mode}")
        st.markdown(f"**Type:** {uploaded_file.type.split('/')[-1].upper()}")
        st.markdown(f"**Confidence Level:** `{confidence}`")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Prediction Summary")
        st.markdown(f"<h4 style='color:{color}; margin-top:-10px'>{label}</h4>", unsafe_allow_html=True)
        st.markdown(f"**Confidence Score:** `{avg_score:.2f}%`")
        st.markdown(f"**Model Disagreement (Variance):** `{variance}`")
        st.markdown(f"**Interpretation:** {interpretation}")
        st.progress(int(avg_score))
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Detection Confidence Gauge")
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=avg_score,
        title={'text': "Confidence (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 60], 'color': "#dff9fb"},
                {'range': [60, 85], 'color': "#ffeaa7"},
                {'range': [85, 100], 'color': "#fab1a0"}
            ],
        }
    ))
    gauge.update_layout(height=250)
    st.plotly_chart(gauge, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Model Score Visualizations")
    df = pd.DataFrame(scores_dict.items(), columns=["Model", "Score"])

    c1, c2 = st.columns(2)
    with c1:
        bar_fig = px.bar(df, x="Model", y="Score", color="Score",
                         color_continuous_scale="Blues", range_y=[0, 100], height=300)
        st.plotly_chart(bar_fig, use_container_width=True)

    with c2:
        line_fig = px.line(df, x="Model", y="Score", markers=True,
                           line_shape="spline", color_discrete_sequence=["#2c3e50"])
        line_fig.update_layout(yaxis_range=[0, 100])
        st.plotly_chart(line_fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Download Report")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download CSV", csv, "phishshield_report.csv", "text/csv")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<div class='footer'>© 2025 PhishShield | Final Year Project | Built with PyTorch & Streamlit</div>", unsafe_allow_html=True)
