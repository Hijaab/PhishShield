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

def get_risk_label(score):
    if score >= 85:
        return "High Confidence"
    elif score >= 60:
        return "Moderate Confidence"
    else:
        return "Low Confidence"

# ----- Page Setup -----
st.set_page_config(page_title="PhishShield", layout="wide")

st.markdown("""
    <style>
    .main-header {
        text-align: center;
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        font-size: 1.1rem;
        color: #607d8b;
        margin-bottom: 2rem;
    }
    .card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 20px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.07);
        margin-bottom: 2rem;
    }
    .center-footer {
        text-align: center;
        font-size: 14px;
        color: #9e9e9e;
        padding-top: 40px;
        padding-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ----- Headers -----
st.markdown("<div class='main-header'>PhishShield – Steganography Detection</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>AI-powered ensemble detection of hidden content in digital images</div>", unsafe_allow_html=True)

# ----- Upload -----
uploaded_file = st.file_uploader("📤 Upload an image (PNG, JPG, JPEG)", type=list(ALLOWED_EXTENSIONS))

if uploaded_file:
    img_tensor, display_image = preprocess_image(uploaded_file)

    with st.spinner("Running model predictions..."):
        predictions, scores = [], {}
        with torch.no_grad():
            for idx, model in enumerate(models):
                output = model(img_tensor)
                score = output.item()
                predictions.append(score)
                scores[f"Model {idx+1}"] = round(score * 100, 2)

        avg_score = round(np.mean(predictions) * 100, 2)
        variance = round(np.var(predictions) * 10000, 2)
        result = "Stego" if avg_score >= 60 else "Non-Steg"
        result_color = '#e53935' if result == 'Stego' else '#43a047'
        confidence_level = get_risk_label(avg_score)
        interpretation = "🔐 Hidden content likely present." if result == "Stego" else "✅ No hidden content detected."

    left_col, right_col = st.columns([1, 1.5], gap="large")

    # ----- LEFT COLUMN -----
    with left_col:
        with st.container():
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("📷 Uploaded Image")
            st.image(display_image, use_container_width=True)
            st.markdown(f"**Size:** {display_image.size[0]} x {display_image.size[1]}  \n"
                        f"**Mode:** {display_image.mode}  \n"
                        f"**Format:** {uploaded_file.type.split('/')[-1].upper()}")
            with st.expander("📊 Raw Model Scores"):
                st.json(scores)
            st.markdown("</div>", unsafe_allow_html=True)

    # ----- RIGHT COLUMN -----
    with right_col:
        with st.container():
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Prediction Summary")
            st.markdown(f"<h4 style='color:{result_color}'>{result}</h4>", unsafe_allow_html=True)
            st.markdown(f"**Confidence Score**: `{avg_score:.2f}%`")
            st.markdown(f"**Confidence Level**: `{confidence_level}`")
            st.markdown(f"**Variance (Disagreement)**: `{variance:.2f}`")
            st.markdown(f"**Interpretation**: {interpretation}")
            st.progress(int(avg_score))
            col1, col2 = st.columns(2)
            col1.metric("Models Used", f"{len(models)}")
            col2.metric("Threshold", "60%")
            st.markdown("</div>", unsafe_allow_html=True)

        with st.container():
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Detection Gauge")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=avg_score,
                title={'text': "Confidence (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': result_color},
                    'steps': [
                        {'range': [0, 60], 'color': "#c8e6c9"},
                        {'range': [60, 85], 'color': "#fff9c4"},
                        {'range': [85, 100], 'color': "#ffcdd2"},
                    ],
                }
            ))
            fig_gauge.update_layout(height=250)
            st.plotly_chart(fig_gauge, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # ----- VISUALIZATIONS -----
    df_scores = pd.DataFrame(scores.items(), columns=["Model", "Score"])

    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Visualizations")
        col1, col2 = st.columns(2)
        with col1:
            bar = px.bar(df_scores, x="Model", y="Score", color="Score",
                         color_continuous_scale="Blues", range_y=[0, 100])
            st.plotly_chart(bar, use_container_width=True)
        with col2:
            line = px.line(df_scores, x="Model", y="Score", markers=True,
                           line_shape="spline", color_discrete_sequence=["#3949ab"])
            line.update_layout(yaxis_range=[0, 100])
            st.plotly_chart(line, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            box = px.box(df_scores, y="Score", points="all", color_discrete_sequence=["#ab47bc"])
            box.update_layout(height=300)
            st.plotly_chart(box, use_container_width=True)
        with col4:
            pie = px.pie(df_scores, names="Model", values="Score",
                         color_discrete_sequence=px.colors.sequential.RdBu)
            pie.update_traces(textinfo="percent+label")
            st.plotly_chart(pie, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ----- DOWNLOAD REPORT -----
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Download Report")
        report_csv = df_scores.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV", report_csv, "phishshield_report.csv", "text/csv")
        st.markdown("</div>", unsafe_allow_html=True)

# ----- FOOTER -----
st.markdown("<div class='center-footer'>© 2025 PhishShield – Final Year Project | Built with PyTorch & Streamlit</div>", unsafe_allow_html=True)
