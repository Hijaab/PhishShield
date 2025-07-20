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

# ----------------- SETUP -----------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# ----------------- MODEL -----------------
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

# ----------------- UTILS -----------------
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

# ----------------- STREAMLIT LAYOUT -----------------
st.set_page_config(page_title="PhishShield", layout="wide")

st.markdown("<h1 style='margin-bottom: 0;'>PhishShield – Steganography Detection</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #6c757d;'>An ensemble deep learning system to detect steganography in images.</p>", unsafe_allow_html=True)
st.divider()

left_col, right_col = st.columns([1, 1.5], gap="large")

with left_col:
    uploaded_file = st.file_uploader("Upload Image", type=list(ALLOWED_EXTENSIONS))
    if uploaded_file:
        img_tensor, display_image = preprocess_image(uploaded_file)
        st.image(display_image, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Running analysis..."):
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
            confidence_level = get_risk_label(avg_score)
            result_color = '#c0392b' if result == "Stego" else '#27ae60'
            interpretation = "Potential hidden content detected." if result == "Stego" else "No hidden content detected."

        with st.expander("Model Scores (Individual Outputs)"):
            st.json(scores)

with right_col:
    if uploaded_file:
        st.subheader("Prediction Summary")
        st.markdown(f"<h3 style='color:{result_color}; margin-top: 0;'>{result}</h3>", unsafe_allow_html=True)
        st.markdown(f"**Confidence Score:** {avg_score:.2f}%")
        st.markdown(f"**Prediction Confidence Level:** `{confidence_level}`")
        st.markdown(f"**Model Disagreement (Variance):** `{variance:.2f}`")
        st.markdown(f"**Interpretation:** {interpretation}")
        st.progress(int(avg_score))

        col1, col2 = st.columns(2)
        col1.metric("Models Used", f"{len(models)}")
        col2.metric("Threshold", "60%")

        st.divider()
        st.subheader("Confidence Gauge")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=avg_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Detection Confidence"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': result_color},
                'steps': [
                    {'range': [0, 60], 'color': "#ecf0f1"},
                    {'range': [60, 85], 'color': "#f9e79f"},
                    {'range': [85, 100], 'color': "#fadbd8"},
                ],
            }
        ))
        fig_gauge.update_layout(height=250)
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.subheader("Score Distribution")
        df_scores = pd.DataFrame(scores.items(), columns=["Model", "Score"])
        bar_chart = px.bar(
            df_scores, x="Model", y="Score", color="Score", 
            color_continuous_scale=px.colors.sequential.Teal, 
            range_y=[0, 100], height=300
        )
        bar_chart.update_layout(template="simple_white", showlegend=False)
        st.plotly_chart(bar_chart, use_container_width=True)

        st.subheader("Score Trend")
        trend_df = pd.DataFrame({
            "Model": [f"Model {i+1}" for i in range(len(predictions))],
            "Score": [round(p * 100, 2) for p in predictions]
        })
        line_chart = px.line(trend_df, x="Model", y="Score", markers=True)
        line_chart.update_layout(template="simple_white", yaxis_range=[0, 100])
        st.plotly_chart(line_chart, use_container_width=True)

        st.subheader("Download Results")
        report_csv = df_scores.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", report_csv, "phishshield_report.csv", "text/csv")

# ----------------- FOOTER -----------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #999999; font-size: 14px; padding-top: 10px;'>
    © 2025 PhishShield – Final Year Project<br>
    Built with PyTorch & Streamlit
</div>
""", unsafe_allow_html=True)
