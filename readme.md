# PhishShield

PhishShield is an AI-powered phishing detection platform that helps identify potentially malicious websites using deep learning techniques. The application analyzes website screenshots and predicts whether a webpage is legitimate or part of a phishing campaign.

## Live Demo

Streamlit Application: https://phishshield-9cks3vxy7habcm8uurgjer.streamlit.app/

## Overview

Phishing attacks remain one of the most common cybersecurity threats, targeting users through fraudulent websites designed to steal credentials, financial information, and personal data.

PhishShield leverages EfficientNet-based deep learning models to classify website screenshots and assist users in identifying suspicious webpages.

## Features

* AI-powered phishing website detection
* Deep learning classification using EfficientNet
* User-friendly Streamlit web interface
* Real-time prediction results
* Multiple trained model checkpoints
* Deployable on cloud platforms

## Technology Stack

### Machine Learning

* Python
* PyTorch
* EfficientNet

### Web Application

* Streamlit
* HTML
* CSS
* JavaScript

### Deployment

* Streamlit Cloud
* Render Configuration Support

## Project Structure

```text
PhishShield/
│
├── model/
│   ├── efficientnet_model_1.pt
│   ├── efficientnet_model_2.pt
│   ├── efficientnet_model_3.pt
│   └── efficientnet_model_4.pt
│
├── static/
│   ├── script.js
│   └── style.css
│
├── templates/
│   └── index.html
│
├── dataset/
│
├── phishshield_streamlit.py
├── requirements.txt
├── packages.txt
├── render.yaml
└── README.md
```

## How It Works

1. User uploads or provides a website screenshot.
2. The image is processed and prepared for inference.
3. The EfficientNet model analyzes visual phishing indicators.
4. The application predicts whether the webpage is:

   * Legitimate
   * Phishing
5. Results are displayed through the Streamlit interface.

## Installation

Clone the repository:

```bash
git clone https://github.com/Hijaab/PhishShield.git
cd PhishShield
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run phishshield_streamlit.py
```

## Future Enhancements

* URL-based phishing analysis
* Domain reputation checking
* WHOIS integration
* Explainable AI predictions
* Browser extension integration
* Real-time threat intelligence feeds

## Learning Outcomes

This project demonstrates:

* Deep Learning Model Development
* Cybersecurity Applications of AI
* Streamlit Application Development
* Model Deployment and Inference
* Security-focused Software Engineering

## Author

Hijaab Sikander

Cybersecurity | DevSecOps | AI Security

LinkedIn: https://linkedin.com/in/hijaabsikander
