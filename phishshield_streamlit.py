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

# ----- Preprocessing -----
def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file).convert('RGB')
    image = image.resize((512, 512))
    img = np.array(image) / 255.0
    img = np.transpose(img, (2, 0, 1)).astype(np.float32)
    img_tensor = torch.tensor(img).unsqueeze(0).to(device)
    return img_tensor, image

# ----- Risk Label -----
def get_risk_label(score):
    if score >= 85:
        return "High Confidence"
    elif score >= 60:
        return "Moderate Confidence"
    else:
        return "Low Confidence"

# ----- UI Setup -----
st.set_page_config(page_title="PhishShield", layout="wide")

# CSS Styling
st.markdown("""
    <style>
        html, body, [class*="css"] {
            font-family: 'Segoe UI', sans-serif;
            background-color: #f4f6f8;
        }
        .main-header {
            text-align: center;
            font-size: 2.8rem;
            font-weight: 700;
            color: #2c3e50;
            margin-top: 1.5rem;
            margin-bottom: 0.3rem;
        }
        .sub-header {
            text-align: center;
            font-size: 1.1rem;
            color: #5d6d7e;
            margin-bottom: 2rem;
        }
        .card-container {
            background-color: #ffffff;
            padding: 1.5rem 2rem;
            border-radius: 20px;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.06);
            margin-bottom: 2rem;
        }
        .center-footer {
            text-align: center;
            font-size: 14px;
            color: #95a5a6;
            margin-top: 3rem;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-header'>PhishShield – Steganography Detection</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>AI-powered detection of hidden content in images</div>", unsafe_allow_html=True)

# ----- Upload Image -----
uploaded_file = st.file_uploader("Upload an image (PNG, JPG, JPEG)", type=list(ALLOWED_EXTENSIONS))

if uploaded_file:
    img_tensor, display_image = preprocess_image(uploaded_file)

    with st.spinner("Running AI detection..."):
        predictions = []
        scores = {}
        with torch.no_grad():
            for idx, model in enumerate(models):
                output = model(img_tensor)
                score = output.item()
                predictions.append(score)
                scores[f"Model {idx+1}"] = round(score * 100, 2)

        avg_score = round(np.mean(predictions) * 100, 2)
        variance = round(np.var(predictions) * 10000, 2)
        result = "Stego" if avg_score >= 60 else "Non-Steg"
        interpretation = "Hidden content likely present." if result == "Stego" else "No hidden content detected."
        confidence_level = get_risk_label(avg_score)
        result_color = '#e74c3c' if result == 'Stego' else '#2ecc71'

        badge_colors = {
            "Low Confidence": "#7bed9f",
            "Moderate Confidence": "#f9ca24",
            "High Confidence": "#ff7675"
        }

        left_col, right_col = st.columns([1, 1.5], gap="large")

        with left_col:
            with st.container():
                st.markdown('<div class="card-container">', unsafe_allow_html=True)
                st.subheader("Uploaded Image")
                st.image(display_image, use_container_width=True)
                st.markdown(f"**Size:** {display_image.size[0]} x {display_image.size[1]}  ")
                st.markdown(f"**Mode:** {display_image.mode}  ")
                st.markdown(f"**Format:** {uploaded_file.type.split('/')[-1].upper()}")
                st.markdown(f"**Confidence Level:** `{confidence_level}`")
                st.markdown('</div>', unsafe_allow_html=True)

        with right_col:
            with st.container():
                st.markdown('<div class="card-container">', unsafe_allow_html=True)
                st.subheader("Prediction Summary")
                st.markdown(f"<h4 style='color:{result_color}; margin-top: -10px;'>{result}</h4>", unsafe_allow_html=True)
                st.markdown(f"**Confidence Score**: `{avg_score:.2f}%`")
                st.markdown(f"**Model Disagreement (Variance)**: `{variance:.2f}`")
                st.markdown(f"**Interpretation**: {interpretation}")
                st.progress(int(avg_score))
                st.markdown('</div>', unsafe_allow_html=True)

        with st.container():
            st.markdown('<div class="card-container">', unsafe_allow_html=True)
            st.subheader("Confidence Gauge")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=avg_score,
                title={'text': "Detection Confidence (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': result_color},
                    'steps': [
                        {'range': [0, 60], 'color': "#dff9fb"},
                        {'range': [60, 85], 'color': "#ffeaa7"},
                        {'range': [85, 100], 'color': "#fab1a0"},
                    ],
                }
            ))
            fig_gauge.update_layout(height=250)
            st.plotly_chart(fig_gauge, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with st.container():
            st.markdown('<div class="card-container">', unsafe_allow_html=True)
            st.subheader("Model Score Visualizations")
            df_scores = pd.DataFrame(scores.items(), columns=["Model", "Score"])

            col1, col2 = st.columns(2)
            with col1:
                bar_chart = px.bar(df_scores, x="Model", y="Score", color="Score",
                                   color_continuous_scale="Blues", range_y=[0, 100], height=300)
                st.plotly_chart(bar_chart, use_container_width=True)

            with col2:
                line_chart = px.line(df_scores, x="Model", y="Score", markers=True,
                                     line_shape="spline", color_discrete_sequence=["#2c3e50"])
                line_chart.update_layout(yaxis_range=[0, 100])
                st.plotly_chart(line_chart, use_container_width=True)

            st.markdown('</div>', unsafe_allow_html=True)

        with st.container():
            st.markdown('<div class="card-container">', unsafe_allow_html=True)
            st.subheader("Download CSV Report")
            report_csv = df_scores.to_csv(index=False).encode('utf-8')
            st.download_button("Download Report", report_csv, "phishshield_report.csv", "text/csv")
            st.markdown('</div>', unsafe_allow_html=True)

# ----- Footer -----
st.markdown("<div class='center-footer'>\u00a9 2025 PhishShield – Final Year Project | Built with PyTorch & Streamlit</div>", unsafe_allow_html=True)
