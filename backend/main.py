import base64
import io
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, UnidentifiedImageError
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
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MOCK_MODE = os.getenv("MOCK_MODE", "false").strip().lower() in {"1", "true", "yes"}
DEVICE = torch.device("cpu")
DEFAULT_CLASSES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]

app = FastAPI(
    title="Weak Boundary Skin Lesion API",
    description=(
        "Research prototype API. Mock mode is for integration testing only; "
        "mock outputs are not model predictions."
    ),
    version="0.2.0",
)
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
class_names = DEFAULT_CLASSES
input_size = (256, 256)
checkpoint_epoch = None
model_load_error = None


@app.on_event("startup")
def load_classification_model() -> None:
    """Load the real classifier when available; allow explicit mock-mode startup."""
    global model, class_names, input_size, checkpoint_epoch, model_load_error

    if MOCK_MODE:
        model = None
        model_load_error = None
        return

    if not MODEL_PATH.is_file():
        model_load_error = f"Model checkpoint not found: {MODEL_PATH}"
        return

    try:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        class_names = checkpoint["class_names"]
        input_size = tuple(checkpoint["input_size"])
        checkpoint_epoch = int(checkpoint["epoch"])
        model = SkinLesionClassifier(num_classes=int(checkpoint["num_classes"]))
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(DEVICE)
        model.eval()
        model_load_error = None
    except Exception as exc:
        model = None
        model_load_error = f"Could not load classification checkpoint: {exc}"


def _read_image(content: bytes) -> Image.Image:
    try:
        with Image.open(io.BytesIO(content)) as source:
            source.verify()
        with Image.open(io.BytesIO(content)) as source:
            return source.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid supported image.",
        ) from exc


async def _read_upload(file: UploadFile) -> tuple[bytes, Image.Image]:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Image is too large. Maximum size is 10 MB.",
        )
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")

    return content, _read_image(content)


def _png_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _mock_classification(filename: str) -> dict[str, Any]:
    # Fixed deterministic scores: useful only for checking the frontend contract.
    values = [0.31, 0.24, 0.15, 0.10, 0.08, 0.07, 0.05]
    scores = [
        {"code": name, "score": value, "percentage": round(value * 100, 2)}
        for name, value in zip(DEFAULT_CLASSES, values)
    ]
    return {
        "filename": os.path.basename(filename or "uploaded_image"),
        "predicted_class": DEFAULT_CLASSES[0],
        "model_score": values[0],
        "model_score_percentage": 31.0,
        "scores": scores,
        "model_epoch": None,
        "input_size": list(input_size),
        "model_version": "mock-classifier-v1",
        "mode": "mock",
        "notice": "MOCK OUTPUT FOR INTEGRATION TESTING ONLY. Not a model prediction or medical advice.",
    }


@app.get("/api/health")
def health() -> dict[str, Any]:
    real_model_loaded = model is not None
    return {
        "status": "ok" if (MOCK_MODE or real_model_loaded) else "degraded",
        "mock_mode": MOCK_MODE,
        "classification_model_loaded": real_model_loaded,
        "segmentation_model_loaded": False,
        "classification_model_version": (
            "mock-classifier-v1" if MOCK_MODE else ("classification-checkpoint" if real_model_loaded else None)
        ),
        "model_epoch": checkpoint_epoch,
        "classes": class_names,
        "model_load_error": model_load_error,
    }


@app.get("/api/models")
def list_models() -> dict[str, Any]:
    return {
        "models": [
            {
                "id": "classification",
                "task": "seven-class image classification",
                "status": "mock" if MOCK_MODE else ("ready" if model is not None else "unavailable"),
                "version": "mock-classifier-v1" if MOCK_MODE else ("classification-checkpoint" if model is not None else None),
            },
            {
                "id": "segmentation",
                "task": "lesion segmentation",
                "status": "mock" if MOCK_MODE else "unavailable",
                "version": "mock-ellipse-v1" if MOCK_MODE else None,
                "note": "No trained segmentation model is connected in the current repository baseline.",
            },
        ],
        "mock_mode": MOCK_MODE,
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, Any]:
    started = time.perf_counter()
    _, image = await _read_upload(file)

    if MOCK_MODE:
        result = _mock_classification(file.filename or "uploaded_image")
        result["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return result

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Classification model is unavailable. Check the checkpoint path or enable MOCK_MODE=true for integration testing.",
        )

    resized = image.resize((input_size[1], input_size[0]), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous().unsqueeze(0)

    try:
        with torch.no_grad():
            logits = model(tensor.to(DEVICE))
            probabilities = torch.softmax(logits, dim=1)[0]
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Classification inference failed.") from exc

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
        "model_score": round(float(probabilities[predicted_index].item()), 6),
        "model_score_percentage": round(float(probabilities[predicted_index].item()) * 100, 2),
        "scores": scores,
        "model_epoch": checkpoint_epoch,
        "input_size": list(input_size),
        "model_version": "classification-checkpoint",
        "mode": "real",
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "notice": (
            "Research prototype only. Not a medical diagnostic tool. "
            "Model scores are not calibrated clinical probabilities."
        ),
    }


@app.post("/api/segment")
async def segment(file: UploadFile = File(...)) -> dict[str, Any]:
    started = time.perf_counter()
    _, image = await _read_upload(file)

    if not MOCK_MODE:
        raise HTTPException(
            status_code=503,
            detail=(
                "No trained segmentation model is configured. Set MOCK_MODE=true "
                "to test frontend integration with a clearly labelled synthetic mask."
            ),
        )

    # Synthetic ellipse for integration only; this is not trained-model inference.
    width, height = image.size
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse(
        (int(width * 0.28), int(height * 0.24), int(width * 0.72), int(height * 0.76)),
        fill=255,
    )

    overlay = image.copy().convert("RGBA")
    tint = Image.new("RGBA", image.size, (255, 0, 0, 0))
    tint.putalpha(mask.point(lambda value: int(value * 0.28)))
    overlay = Image.alpha_composite(overlay, tint).convert("RGB")

    foreground_ratio = round(float(np.asarray(mask).mean() / 255.0 * 100), 2)
    return {
        "filename": os.path.basename(file.filename or "uploaded_image"),
        "mask": _png_data_url(mask),
        "overlay": _png_data_url(overlay),
        "foreground_ratio": foreground_ratio,
        "input_size": [height, width],
        "threshold": 0.5,
        "model_version": "mock-ellipse-v1",
        "mode": "mock",
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "notice": (
            "SYNTHETIC MOCK MASK FOR INTEGRATION TESTING ONLY. "
            "This is not MALUNet output and must not be used for medical interpretation."
        ),
    }


@app.get("/api/evaluate")
def evaluate_status() -> dict[str, Any]:
    """Report evaluation availability without inventing metric values."""
    return {
        "status": "not_ready",
        "available": False,
        "metrics": None,
        "message": (
            "No evaluation result artifact is connected to the API yet. "
            "Run the project evaluation scripts and define a versioned result format first."
        ),
    }
