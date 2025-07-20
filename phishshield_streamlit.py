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

# ----- Utility -----
def get_risk_label(score):
    if score >= 85:
        return "High Confidence"
    elif score >= 60:
        return "Moderate Confidence"
    else:
        return "Low Confidence"

# ----- Streamlit UI -----
st.set_page_config(page_title="PhishShield", layout="wide")
st.title("PhishShield – Steganography Detection")
st.caption("Ensemble-based detection of hidden content in digital images.")
st.markdown("---")

uploaded_file = st.file_uploader("Upload an image", type=list(ALLOWED_EXTENSIONS))
left_col, right_col = st.columns([1, 1.5])

if uploaded_file:
    with st.spinner("Running analysis..."):
        img_tensor, display_image = preprocess_image(uploaded_file)
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
        result_color = '#ff6b6b' if result == 'Stego' else '#4caf50'
        interpretation = "Potential hidden content detected." if result == "Stego" else "No hidden content detected."
        confidence_level = get_risk_label(avg_score)

        # ----- LEFT COLUMN -----
        with left_col:
            st.subheader("Uploaded Image")
            st.image(display_image, use_container_width=True)

            # Image Info Card
            st.markdown("### Image Info")
            st.markdown(f"- **Size**: {display_image.size[0]} x {display_image.size[1]}")
            st.markdown(f"- **Mode**: {display_image.mode}")
            st.markdown(f"- **Format**: {uploaded_file.type.split('/')[-1].upper()}")

            # Risk Label Badge
            st.markdown("---")
            risk_colors = {
                "Low Confidence": "#81c784",
                "Moderate Confidence": "#ffd54f",
                "High Confidence": "#e57373"
            }
            st.markdown(
                f"<div style='background-color:{risk_colors[confidence_level]};"
                f"color:#000;padding:8px 16px;border-radius:12px;text-align:center;"
                f"font-weight:600;font-size:16px;'>"
                f"Risk Category: {confidence_level}"
                f"</div>",
                unsafe_allow_html=True
            )

            st.markdown("---")
            with st.expander("Model Raw Scores"):
                st.json(scores)

        # ----- RIGHT COLUMN -----
        with right_col:
            st.subheader("Prediction Summary")
            st.markdown(f"<h4 style='color:{result_color}'>{result}</h4>", unsafe_allow_html=True)
            st.markdown(f"**Confidence Score**: {avg_score:.2f}%")
            st.markdown(f"**Prediction Confidence Level**: `{confidence_level}`")
            st.markdown(f"**Model Disagreement (Variance)**: `{variance:.2f}`")
            st.markdown(f"**Interpretation**: {interpretation}")
            st.progress(int(avg_score))

            col1, col2 = st.columns(2)
            col1.metric("Models Used", f"{len(models)}")
            col2.metric("Threshold", "60%")

            st.markdown("---")
            st.subheader("Detection Confidence Gauge")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=avg_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Confidence %"},
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

            df_scores = pd.DataFrame(scores.items(), columns=["Model", "Score"])

            st.subheader("Model Score Distribution")
            bar_chart = px.bar(df_scores, x="Model", y="Score", color="Score",
                               color_continuous_scale="Viridis", range_y=[0, 100], height=300)
            bar_chart.update_layout(template="simple_white", showlegend=False)
            st.plotly_chart(bar_chart, use_container_width=True)

            st.subheader("Prediction Trend Line")
            trend_df = pd.DataFrame({
                "Model": [f"Model {i+1}" for i in range(len(predictions))],
                "Score": [round(p * 100, 2) for p in predictions]
            })
            line_chart = px.line(trend_df, x="Model", y="Score", markers=True,
                                 line_shape="spline", color_discrete_sequence=["#00796b"])
            line_chart.update_layout(template="plotly_white", yaxis_range=[0, 100])
            st.plotly_chart(line_chart, use_container_width=True)

            st.subheader("Confidence Spread – Box Plot")
            box = px.box(df_scores, y="Score", points="all", color_discrete_sequence=["#8e24aa"])
            box.update_layout(template="plotly_white", height=300)
            st.plotly_chart(box, use_container_width=True)

            st.subheader("Confidence Composition – Pie Chart")
            pie = px.pie(df_scores, names="Model", values="Score", color_discrete_sequence=px.colors.sequential.RdBu)
            pie.update_traces(textinfo="percent+label")
            st.plotly_chart(pie, use_container_width=True)

            st.markdown("---")
            st.subheader("Download CSV Report")
            report_csv = df_scores.to_csv(index=False).encode('utf-8')
            st.download_button("Download Report", report_csv, "phishshield_report.csv", "text/csv")

# ----- Footer -----
st.markdown("""
<style>
footer {visibility: hidden;}
.stApp {
    background-color: #f9f9f9;
    font-family: 'Segoe UI', sans-serif;
}
h1, h2, h3, h4 {
    font-weight: 600;
}
.center-footer {
    text-align: center;
    font-size: 14px;
    margin-top: 50px;
    color: #666;
}
</style>
""", unsafe_allow_html=True)

# Centered footer text
st.markdown("<div class='center-footer'>© 2025 PhishShield – Final Year Project | Built with PyTorch & Streamlit</div>", unsafe_allow_html=True)
