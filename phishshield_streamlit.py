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

# ----- Model -----
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

def get_risk_label(score):
    if score >= 85:
        return "High Confidence"
    elif score >= 60:
        return "Moderate Confidence"
    else:
        return "Low Confidence"

# ----- Page Settings -----
st.set_page_config(page_title="PhishShield", layout="wide")

# ----- Custom CSS -----
st.markdown("""
    <style>
        .main-title {
            text-align: center;
            font-size: 2.5rem;
            font-weight: bold;
            color: #2c3e50;
            margin-top: 0.5rem;
            margin-bottom: 0.2rem;
        }
        .subtitle {
            text-align: center;
            font-size: 1.1rem;
            color: #5d6d7e;
            margin-bottom: 2rem;
        }
        .stContainer {
            background-color: #ffffff;
            padding: 1.5rem 2rem;
            border-radius: 18px;
            box-shadow: 0 6px 18px rgba(0, 0, 0, 0.05);
            margin-bottom: 2rem;
        }
        .footer {
            text-align: center;
            font-size: 13px;
            margin-top: 3rem;
            color: #888;
        }
    </style>
""", unsafe_allow_html=True)

# ----- Heading -----
st.markdown("<div class='main-title'>PhishShield – Steganography Detection</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI-powered detection of hidden content in images</div>", unsafe_allow_html=True)

# ----- File Upload -----
uploaded_file = st.file_uploader("Upload an image", type=list(ALLOWED_EXTENSIONS))

if uploaded_file:
    img_tensor, display_image = preprocess_image(uploaded_file)

    with st.spinner("Analyzing..."):
        predictions, scores = [], {}
        with torch.no_grad():
            for idx, model in enumerate(models):
                score = model(img_tensor).item()
                predictions.append(score)
                scores[f"Model {idx+1}"] = round(score * 100, 2)

        avg_score = round(np.mean(predictions) * 100, 2)
        variance = round(np.var(predictions) * 10000, 2)
        result = "Stego" if avg_score >= 80 else "Non-Steg"
        interpretation = "Potential hidden content detected." if result == "Stego" else "No hidden content detected."
        confidence_level = get_risk_label(avg_score)
        result_color = "#e74c3c" if result == "Stego" else "#2ecc71"

        risk_colors = {
            "Low Confidence": "#a2d5c6",
            "Moderate Confidence": "#ffeaa7",
            "High Confidence": "#fab1a0"
        }

        # ----- Layout Columns -----
        left_col, right_col = st.columns([1, 1.5], gap="large")

        with left_col:
            with st.container():
                st.subheader("Uploaded Image")
                st.image(display_image, use_container_width=True)
                st.markdown(f"- **Size**: {display_image.size[0]} x {display_image.size[1]}")
                st.markdown(f"- **Mode**: {display_image.mode}")
                st.markdown(f"- **Format**: {uploaded_file.type.split('/')[-1].upper()}")
                st.markdown(f"<div style='background-color:{risk_colors[confidence_level]};"
                            f"padding: 10px 14px; border-radius: 12px; margin-top: 15px; "
                            f"text-align:center; font-weight:600;'>"
                            f"Risk Category: {confidence_level}</div>", unsafe_allow_html=True)

        with right_col:
            with st.container():
                st.subheader("Prediction Summary")
                st.markdown(f"<h4 style='color:{result_color}; margin-top:-10px;'>{result}</h4>", unsafe_allow_html=True)
                st.markdown(f"- **Confidence Score**: `{avg_score:.2f}%`")
                st.markdown(f"- **Variance**: `{variance:.2f}`")
                st.markdown(f"- **Confidence Level**: `{confidence_level}`")
                st.markdown(f"- **Interpretation**: {interpretation}")
                st.progress(int(avg_score))

        # ----- Gauge -----
        with st.container():
            st.subheader("Detection Confidence Gauge")
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=avg_score,
                title={'text': "Confidence %"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': result_color},
                    'steps': [
                        {'range': [0, 60], 'color': "#dff9fb"},
                        {'range': [60, 85], 'color': "#ffeaa7"},
                        {'range': [85, 100], 'color': "#fab1a0"},
                    ]
                }
            ))
            gauge.update_layout(height=250)
            st.plotly_chart(gauge, use_container_width=True)

        # ----- Visualizations -----
        df_scores = pd.DataFrame(scores.items(), columns=["Model", "Score"])

        vis_cols = st.columns(2)

        with vis_cols[0]:
            with st.container():
                st.subheader("Bar Chart")
                bar = px.bar(df_scores, x="Model", y="Score", color="Score", color_continuous_scale="Blues", range_y=[0, 100])
                st.plotly_chart(bar, use_container_width=True)

        with vis_cols[1]:
            with st.container():
                st.subheader("Line Chart")
                line = px.line(df_scores, x="Model", y="Score", markers=True, line_shape="spline",
                               color_discrete_sequence=["#34495e"])
                line.update_layout(yaxis_range=[0, 100])
                st.plotly_chart(line, use_container_width=True)

        col3, col4 = st.columns(2)

        with col3:
            with st.container():
                st.subheader("Box Plot")
                box = px.box(df_scores, y="Score", points="all", color_discrete_sequence=["#8e24aa"])
                st.plotly_chart(box, use_container_width=True)

        with col4:
            with st.container():
                st.subheader("Pie Chart")
                pie = px.pie(df_scores, names="Model", values="Score", color_discrete_sequence=px.colors.sequential.RdBu)
                pie.update_traces(textinfo="percent+label")
                st.plotly_chart(pie, use_container_width=True)

        # ----- Download -----
        with st.container():
            st.subheader("Download CSV Report")
            csv = df_scores.to_csv(index=False).encode("utf-8")
            st.download_button("Download Report", csv, "phishshield_report.csv", "text/csv")

# ----- Footer -----
st.markdown("<div class='footer'>© 2025 PhishShield – Final Year Project | Built with PyTorch & Streamlit</div>", unsafe_allow_html=True)
