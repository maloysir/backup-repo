import os
import numpy as np
import torch
import torch.nn as nn                 
import torch.nn.functional as F
from PIL import Image
import pickle
from torchvision import transforms
from torchvision.models import resnet50
import io

# ✅ ADD FASTAPI
from fastapi import FastAPI, File, UploadFile

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- CONFIG ----
MODEL_PATH = "best_resnet50.pth"
DEVICE = torch.device("cpu")

# ---- CLASSES ----
CLASSES = [
    "biotite","bornite","chrysocolla",
    "malachite","muscovite","pyrite","quartz"
]

# ---- TRANSFORM ----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---- MODEL LOADING ----
def load_model():
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# Feature extractor
feature_extractor = nn.Sequential(*list(model.children())[:-1])
feature_extractor.to(DEVICE)
feature_extractor.eval()

# ---- LOAD OOD ----
with open("ood_stats_resnet.pkl", "rb") as f:
    stats = pickle.load(f)

means = stats["means"]
inv_cov = stats["inv_cov"]
threshold = stats["threshold"]

# ============================================================
# ✅ FASTAPI ENDPOINT (ONLY ADDITION)
# ============================================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # ✅ FIX: DEFINE image
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")

    # ---- PREPROCESS ----
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        
        outputs = model(img_tensor)

        # ---- CLASSIFICATION ----
        probs = F.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()

        # ---- FEATURE EXTRACTION ----
        feat = feature_extractor(img_tensor)
        feat = feat.view(feat.size(0), -1)
        feat = feat.cpu().numpy().flatten()

    # ---- NORMALIZATION ----
    feat = feat / (np.linalg.norm(feat) + 1e-8)

    # ---- COSINE DISTANCE ----
    distance = 1 - np.dot(feat, means[pred])

    # ---- OOD CHECK ----
    if distance > 0.55:

        return {
            "label": "Unknown Mineral",
            "confidence": 0.0,
            "distance": float(distance)
        }

    else:
        probs = F.softmax(outputs, dim=1).squeeze()

        label = CLASSES[pred]
        confidence = probs[pred].item()

        return {
            "label": label,
            "confidence": float(confidence),
            "distance": float(distance)
        }