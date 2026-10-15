
import io
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.classification_model import SkinLesionClassifier


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "classification"
    / "classification_rgb_best_macro_f1.pth"
)

DEVICE = torch.device("cpu")

app = FastAPI(
    title="Skin Lesion Classification API",
    description="Research prototype for seven-class image classification.",
    version="0.1.0",
)

# Allow the local Vite development server to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

model = None
class_names = None
input_size = None
checkpoint_epoch = None


@app.on_event("startup")
def load_classification_model():
    global model, class_names, input_size, checkpoint_epoch

    if not MODEL_PATH.is_file():
        raise RuntimeError(f"Model checkpoint not found: {MODEL_PATH}")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    class_names = checkpoint["class_names"]
    input_size = tuple(checkpoint["input_size"])
    checkpoint_epoch = int(checkpoint["epoch"])

    model = SkinLesionClassifier(
        num_classes=int(checkpoint["num_classes"])
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_epoch": checkpoint_epoch,
        "classes": class_names,
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Classification model is not loaded.",
        )

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload an image file.",
        )

    # Limit uploads to 10 MB.
    max_bytes = 10 * 1024 * 1024
    content = await file.read(max_bytes + 1)

    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail="Image is too large. Maximum size is 10 MB.",
        )

    try:
        with Image.open(io.BytesIO(content)) as img:
            image = img.convert("RGB")
            image = image.resize(
                (input_size[1], input_size[0]),
                Image.Resampling.BILINEAR,
            )
            array = np.asarray(image, dtype=np.float32) / 255.0
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid supported image.",
        )

    tensor = torch.from_numpy(array)
    tensor = tensor.permute(2, 0, 1).contiguous().unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor.to(DEVICE))
        probabilities = torch.softmax(logits, dim=1)[0]

    predicted_index = int(torch.argmax(probabilities).item())

    scores = [
        {
            "code": name,
            "score": round(float(probabilities[i].item()), 6),
            "percentage": round(float(probabilities[i].item()) * 100, 2),
        }
        for i, name in enumerate(class_names)
    ]

    return {
        "filename": os.path.basename(file.filename or "uploaded_image"),
        "predicted_class": class_names[predicted_index],
        "model_score": round(
            float(probabilities[predicted_index].item()), 6
        ),
        "model_score_percentage": round(
            float(probabilities[predicted_index].item()) * 100, 2
        ),
        "scores": scores,
        "model_epoch": checkpoint_epoch,
        "input_size": list(input_size),
        "notice": (
            "Research prototype only. Not a medical diagnostic tool. "
            "Model scores are not calibrated clinical probabilities."
        ),
    }