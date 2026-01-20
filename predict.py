import os
import torch
import torch.nn as nn
from flask import Flask, render_template, request
from torchvision import transforms
from torchvision.models import efficientnet_b0
from PIL import Image

# -----------------------
# APP SETUP
# -----------------------
app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# LOAD MODEL
# -----------------------
MODEL_PATH = "models/fruit_quality_model.pth"

checkpoint = torch.load(MODEL_PATH, map_location=device)
class_names = checkpoint["class_names"]

model = efficientnet_b0(pretrained=False)
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    len(class_names)
)

model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

# -----------------------
# IMAGE TRANSFORM
# -----------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------
# HELPERS
# -----------------------
def clean_label(label):
    return label.replace("_", " ").title()

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, 1)

    label = clean_label(class_names[pred.item()])
    return label, confidence.item()

# -----------------------
# ROUTES
# -----------------------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    image_path = None

    if request.method == "POST":
        file = request.files.get("file")

        if file:
            image_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(image_path)

            prediction, confidence = predict_image(image_path)
            confidence = round(confidence * 100, 2)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image_path=image_path
    )

@app.route("/health")
def health():
    return {"status": "healthy"}, 200

# -----------------------
# RUN
# -----------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
