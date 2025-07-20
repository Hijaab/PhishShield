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
from fpdf import FPDF
import base64
import io

# ----- Setup -----
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# ----- Model Definition -----
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

# ----- Preprocessing -----
def preprocess_image(image):
    image = image.convert('RGB')
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

# ----- Streamlit UI -----
st.set_page_config(page_title="PhishShield", layout="wide")

dark_mode = st.toggle("🌙 Dark Mode")
if dark_mode:
    st.markdown("<style>body { background-color: #0e1117; color: #fafafa; }</style>", unsafe_allow_html=True)

st.title("🛡️ PhishShield – Steganography Detection")
st.caption("Upload an image and detect hidden steganographic content using AI ensemble.")

# Upload
uploaded_file = st.file_uploader("📁 Upload an image", type=list(ALLOWED_EXTENSIONS))

# ----- Prediction -----
if uploaded_file:
    img_tensor, display_image = preprocess_image(Image.open(uploaded_file))
    predictions, scores = [], {}

    with torch.no_grad():
        for idx, model in enumerate(models):
            output = model(img_tensor)
            score = output.item()
            predictions.append(score)
            scores[f"Model {idx+1}"] = round(score * 100, 2)

    avg_score = round(np.mean(predictions) * 100, 2)
    result = "Stego" if avg_score >= 60 else "Non-Steg"
    result_color = '#d9534f' if result == 'Stego' else '#5cb85c'
    confidence_level = get_risk_label(avg_score)

    # Layout: Result Right, Image Left
    left_col, right_col = st.columns([1.5, 1])
    with left_col:
        st.markdown(f"<h3 style='color:{result_color}'>{result}</h3>", unsafe_allow_html=True)
        st.write(f"🔢 Confidence Score: `{avg_score}%`")
        st.write(f"🔒 Confidence Level: `{confidence_level}`")
        st.progress(int(avg_score))

        df_scores = pd.DataFrame(scores.items(), columns=["Model", "Score"])
        st.subheader("📊 Model Scores")
        st.dataframe(df_scores)

        chart = px.bar(df_scores, x="Model", y="Score", color="Score",
                       color_continuous_scale="RdYlGn", height=300)
        st.plotly_chart(chart, use_container_width=True)

        # PDF Report
        if st.button("📄 Export PDF Report"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="PhishShield Report", ln=True, align='C')
            pdf.cell(200, 10, txt=f"Result: {result}", ln=True)
            pdf.cell(200, 10, txt=f"Confidence Score: {avg_score:.2f}%", ln=True)
            pdf.cell(200, 10, txt=f"Confidence Level: {confidence_level}", ln=True)
            pdf.cell(200, 10, txt="Model Scores:", ln=True)
            for model, score in scores.items():
                pdf.cell(200, 10, txt=f"{model}: {score}%", ln=True)

            pdf_output = io.BytesIO()
            pdf.output(pdf_output)
            b64 = base64.b64encode(pdf_output.getvalue()).decode()
            href = f'<a href="data:application/pdf;base64,{b64}" download="phishshield_report.pdf">📥 Download PDF Report</a>'
            st.markdown(href, unsafe_allow_html=True)

    with right_col:
        st.image(display_image, caption="📸 Uploaded Image", use_container_width=True)

# ----- Footer -----
st.markdown("""
<style>
footer {visibility: hidden;}
.stApp {
    position: relative;
    padding-bottom: 80px;
}
footer::after {
    content: "© 2025 PhishShield | Final Year Project | Built with PyTorch + Streamlit";
    visibility: visible;
    display: block;
    position: fixed;
    background: #f8f9fa;
    padding: 10px;
    text-align: center;
    bottom: 0;
    left: 0;
    width: 100%;
    color: #666;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)
