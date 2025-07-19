from flask import Flask, render_template, request, send_from_directory
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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

# ----- Load all models -----
models = []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for i in range(1, 5):
    model = ENSModel().to(device)
    model_path = os.path.join("model", f"efficientnet_model_{i}.pt")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    models.append(model)

# ----- Image Preprocessing -----
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (512, 512))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = np.transpose(img, (2, 0, 1)).astype(np.float32)
    img_tensor = torch.tensor(img).unsqueeze(0).to(device)
    return img_tensor

# ----- Cache Functions -----
def is_cached_result(filename):
    return os.path.exists(f"cache/{filename}.txt")

def save_cache_result(filename, result, avg_score, scores):
    os.makedirs("cache", exist_ok=True)
    with open(f"cache/{filename}.txt", "w") as f:
        f.write(f"{result},{avg_score},{scores}\n")

def load_cache_result(filename):
    try:
        with open(f"cache/{filename}.txt", "r") as f:
            line = f.readline().strip()
            result, avg_score, scores = line.split(",", 2)
            return result, float(avg_score), eval(scores)
    except:
        return None, None, {}

# ----- Routes -----
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            file = request.files['image']
            if file and '.' in file.filename:
                ext = file.filename.rsplit('.', 1)[1].lower()
                if ext in ALLOWED_EXTENSIONS:
                    filename = secure_filename(file.filename)
                    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(path)

                    result, avg_score, scores = None, None, {}

                    if is_cached_result(filename):
                        result, avg_score, scores = load_cache_result(filename)
                    else:
                        img_tensor = preprocess_image(path)
                        predictions = []
                        scores = {}
                        with torch.no_grad():
                            for idx, model in enumerate(models):
                                output = model(img_tensor)
                                pred_score = output.item()
                                predictions.append(pred_score)
                                scores[f"Model {idx+1}"] = round(pred_score * 100, 2)

                        avg_pred = sum(predictions) / len(predictions)
                        result = 'Stego' if avg_pred >= 0.6 else 'Non-Steg'
                        avg_score = round(avg_pred * 100, 2)

                        save_cache_result(filename, result, avg_score, scores)

                    return render_template(
                        'index.html',
                        result=result,
                        score=avg_score,
                        filename=filename,
                        scores=scores
                    )
        except Exception as e:
            print("⚠️ ERROR:", e)
            return render_template('index.html', result="Error processing image.", scores={})
    return render_template('index.html', result=None, scores={})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    import os
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
