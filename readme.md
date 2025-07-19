# Steganography Detector Dashboard

Small Flask web app that wraps a PyTorch EfficientNet model to classify
images as **Stego** or **Clean**.

## Quick Start
```bash
git clone …
cd dashboard‑project
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py
