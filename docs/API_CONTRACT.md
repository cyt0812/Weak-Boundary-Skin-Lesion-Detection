# Backend API Contract (Informal Review)

Base URL: `http://127.0.0.1:8000`

Image endpoints accept multipart form data with a single field named `file`. Maximum upload size is 10 MiB. Errors use FastAPI's JSON `detail` field.

## Endpoints

| Method | Path | Purpose | Expected status |
|---|---|---|---|
| GET | `/api/health` | Service/model readiness | 200; status may be `degraded` |
| GET | `/api/models` | Model availability and versions | 200 |
| POST | `/api/predict` | Seven-class image classification | 200, 400, 413, 503 |
| POST | `/api/segment` | Segmentation output contract | 200 in mock mode; 400, 413, 503 |
| GET | `/api/evaluate` | Evaluation data availability | 200; metrics remain null until real results are connected |

## Start in mock mode

Mock mode runs without model checkpoints. Mock classification returns deterministic sample scores. Mock segmentation creates a synthetic ellipse mask and overlay so the frontend can test its rendering and JSON handling.

**Mock outputs are not model predictions.** Segmentation's `model_version` is `mock-ellipse-v1`, and `mode` is `mock`. Do not describe this as MALUNet inference or use it for medical interpretation.

Windows CMD (from project root):
```bat
conda activate skin-lesion
set MOCK_MODE=true
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

PowerShell:
```powershell
$env:MOCK_MODE = "true"
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Swagger UI: `http://127.0.0.1:8000/docs`

## curl smoke tests

Health:
```bash
curl http://127.0.0.1:8000/api/health
```

Model registry:
```bash
curl http://127.0.0.1:8000/api/models
```

Classification:
```bash
curl -X POST "http://127.0.0.1:8000/api/predict" -F "file=@sample.jpg"
```

Segmentation:
```bash
curl -X POST "http://127.0.0.1:8000/api/segment" -F "file=@sample.jpg"
```

Evaluation availability:
```bash
curl http://127.0.0.1:8000/api/evaluate
```

## Response contract

Classification returns `filename`, `predicted_class`, `model_score`, `model_score_percentage`, seven `scores`, `model_version`, `mode`, `latency_ms`, and a research notice.

Segmentation returns `filename`, `mask`, `overlay` (PNG data URLs), `foreground_ratio`, `input_size`, `threshold`, `model_version`, `mode`, `latency_ms`, and a notice.

Example response values in demos must be labelled illustrative. Do not report mock latency as real model latency or mock scores as evaluation results.

## Error behavior

- 400: unsupported content type, empty upload, or invalid image bytes.
- 413: upload exceeds 10 MiB.
- 503: real model is required but unavailable, or real segmentation has not been integrated.
- 500: unexpected classification inference failure.

## Current limitations

- No trained segmentation model is connected in the current repository baseline; non-mock `POST /api/segment` returns 503.
- `GET /api/evaluate` reports status only. It deliberately does not invent Dice, IoU, Macro-F1, or Accuracy.
- Classification and segmentation run independently. Segmentation-guided fusion is not implemented.
