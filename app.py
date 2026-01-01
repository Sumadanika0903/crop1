import os
from pathlib import Path
import io
from datetime import datetime

from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
from werkzeug.utils import secure_filename

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# ---- Config ----
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
RESULT_FOLDER = BASE_DIR / "static" / "results"
MODEL_PATH = BASE_DIR / "saved_models" / "best_resnet18.pth"   # change if needed
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
RESULT_FOLDER.mkdir(parents=True, exist_ok=True)

# Flask app
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["RESULT_FOLDER"] = str(RESULT_FOLDER)
app.secret_key = "replace-this-with-a-secret-key"  # change for production

# ---- Device ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---- Load model checkpoint ----
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model checkpoint not found at {MODEL_PATH}. Place your checkpoint there.")

ckpt = torch.load(str(MODEL_PATH), map_location="cpu")
# checkpoint should contain model_state_dict and class2idx; fallback handled below
state = ckpt.get("model_state_dict", ckpt)

if "class2idx" in ckpt:
    class2idx = ckpt["class2idx"]
else:
    # fallback: user must adjust classes here if checkpoint has no mapping.
    # Put an example mapping or raise informative error
    raise RuntimeError("Checkpoint does not contain 'class2idx'. Save class2idx with the checkpoint or update code to provide class list.")

idx2class = {v:k for k,v in class2idx.items()}
num_classes = len(idx2class)

# Build model architecture (ResNet18 used during training)
model = models.resnet18(weights=None)
in_feats = model.fc.in_features
model.fc = nn.Linear(in_feats, num_classes)

# Fix state keys if saved from DataParallel
if any(k.startswith("module.") for k in list(state.keys())):
    new_state = {}
    for k,v in state.items():
        new_state[k.replace("module.", "")] = v
    state = new_state

# Load weights
try:
    model.load_state_dict(state)
except Exception as e:
    # try non-strict load
    model.load_state_dict(state, strict=False)
model = model.to(device)
model.eval()

# Preprocess (must match training preprocess)
IMG_SIZE = int(ckpt.get("img_size", 224))
preprocess = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.15)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# ---- Helpers ----
def allowed_file(filename):
    return "." in filename and Path(filename).suffix.lower() in ALLOWED_EXT

def overlay_label(img_pil, label_text):
    """Return a PIL image with a small label overlay (top-left)."""
    img = img_pil.copy().convert("RGBA")
    draw = ImageDraw.Draw(img, "RGBA")
    W, H = img.size
    # font attempt
    try:
        font = ImageFont.truetype("arial.ttf", size=int(H * 0.05))
    except Exception:
        font = ImageFont.load_default()
    # compute text size robustly
    try:
        bbox = draw.textbbox((0,0), label_text, font=font)
        w = bbox[2]-bbox[0]; h = bbox[3]-bbox[1]
    except Exception:
        try:
            w, h = draw.textsize(label_text, font=font)
        except Exception:
            w, h = font.getsize(label_text)
    padding = int(H * 0.02)
    rect_coords = (0, 0, w + 2*padding, h + 2*padding)
    draw.rectangle(rect_coords, fill=(0,0,0,150))
    draw.text((padding, padding), label_text, font=font, fill=(255,255,255,255))
    return img.convert("RGB")

def predict_image(pil_image):
    """Run preprocessing and model inference. Return (label, confidence_score, probs_array)."""
    img_tensor = preprocess(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy().squeeze()
    top_idx = int(np.argmax(probs))
    label = idx2class.get(top_idx, str(top_idx))
    confidence = float(probs[top_idx])
    return label, confidence, probs

# ---- Routes ----
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        flash("No file part")
        return redirect(url_for("index"))
    file = request.files["file"]
    if file.filename == "":
        flash("No selected file")
        return redirect(url_for("index"))
    if not allowed_file(file.filename):
        flash("File type not allowed")
        return redirect(url_for("index"))

    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_name = f"{timestamp}_{filename}"
    saved_path = UPLOAD_FOLDER / saved_name
    file.save(str(saved_path))

    # open image with PIL
    try:
        pil_img = Image.open(saved_path).convert("RGB")
    except Exception as e:
        flash("Failed to read image: " + str(e))
        return redirect(url_for("index"))

    # predict
    label, confidence, _ = predict_image(pil_img)

    # annotated image (overlay label). Save result
    annotated = overlay_label(pil_img, label)
    result_name = f"result_{timestamp}_{filename}"
    result_path = RESULT_FOLDER / result_name
    annotated.save(result_path)

    # render result page showing original and annotated
    return render_template("result.html",
                           orig_img=url_for("static", filename=f"uploads/{saved_name}"),
                           result_img=url_for("static", filename=f"results/{result_name}"),
                           label=label,
                           confidence=f"{confidence:.4f}")

# serve static (Flask already serves /static); optional route to list predictions etc.

if __name__ == "__main__":
    # run in debug for local testing
    app.run(host="0.0.0.0", port=5000, debug=True)
