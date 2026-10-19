# Informal Review: Backend + Integration Test Plan

## Architecture

```mermaid
flowchart LR
  U[React / Vite UI] -->|multipart/form-data: file| API[FastAPI]
  API --> V[Upload validation]
  V --> P[/api/predict]
  V --> S[/api/segment]
  P --> CM[Real classifier checkpoint]
  P -. MOCK_MODE=true .-> MC[Deterministic mock classifier]
  S --> SM[Trained segmentation model: not integrated]
  S -. MOCK_MODE=true .-> MS[Synthetic ellipse mask]
  API --> H[/api/health]
  API --> M[/api/models]
  API --> E[/api/evaluate]
  CM --> R[JSON response]
  MC --> R
  MS --> R
  R --> U
```

Solid paths represent the intended real-model route. Dotted paths are mock-only integration routes. A synthetic mask proves request/response and UI rendering only; it does not prove segmentation quality.

## Current API contract

| Endpoint | Input | Output / purpose | Current status |
|---|---|---|---|
| `GET /api/health` | None | Service and model readiness | Implemented |
| `GET /api/models` | None | Model status and version labels | Implemented |
| `POST /api/predict` | Multipart `file` | Seven class scores, predicted class, version, mode, latency | Real classifier if checkpoint exists; deterministic mock if enabled |
| `POST /api/segment` | Multipart `file` | PNG data URLs for mask and overlay | Synthetic mock only; 503 in real mode until a trained model is integrated |
| `GET /api/evaluate` | None | Evaluation-data readiness | Status only; no metrics are fabricated |

Full contract and example requests: [API_CONTRACT.md](API_CONTRACT.md).

## Integration test matrix

| ID | Scenario | Expected result | Status to record |
|---|---|---|---|
| IT-01 | Health check | 200; mock/real readiness is explicit | Not run in this document |
| IT-02 | Model registry | 200; classifier and segmenter states listed | Not run in this document |
| IT-03 | Valid PNG/JPG to `/api/predict` in mock mode | 200; seven scores; `mode=mock` | Not run in this document |
| IT-04 | Valid PNG/JPG to `/api/segment` in mock mode | 200; mask and overlay data URLs; synthetic notice | Not run in this document |
| IT-05 | Unsupported MIME type | 400 | Not run in this document |
| IT-06 | Corrupt image bytes | 400 | Not run in this document |
| IT-07 | Upload larger than 10 MiB | 413 | Not run in this document |
| IT-08 | Real mode with missing classifier checkpoint | Service starts degraded; prediction returns 503 | Not run in this document |
| IT-09 | Real mode segmentation without trained weights | 503; clear explanation | Not run in this document |
| IT-10 | Evaluation endpoint without metrics file | 200; `available=false`, `metrics=null` | Not run in this document |
| IT-11 | Frontend upload → backend → rendered response | UI displays classification and synthetic mask with mock warning | Not run in this document |
| IT-12 | Response timing | `latency_ms` present; distinguish mock from real inference | Not run in this document |

Do not mark any test as passed until it has actually been executed and the observed result has been recorded.

## How to run

1. Install dependencies from the repository root: `python -m pip install -r requirements.txt`.
2. Start mock backend in Windows CMD:
   ```bat
   set MOCK_MODE=true
   python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
   ```
3. Start frontend in another terminal:
   ```bat
   cd frontend
   npm ci
   npm run dev
   ```
4. Open `http://localhost:5173`, upload a test image, and try classification and segmentation separately.
5. Run backend tests from project root: `python -m pytest -q`.
6. Open Swagger UI at `http://127.0.0.1:8000/docs` and keep screenshots or console output as evidence.

Use a non-sensitive, permitted sample image. Do not use the mock outputs for any clinical interpretation.

## Risks and dependencies

1. **Segmentation model delivery:** no trained MALUNet implementation/checkpoint is present in the currently documented baseline. Need the algorithm teammate to provide model code, weight file, preprocessing, output tensor shape, and license/source attribution.
2. **Classification checkpoint:** ignored by Git; each developer needs an authorized local checkpoint for real inference.
3. **Model output agreement:** agree on input size, normalization, class order, mask threshold, mask dimensions, and model version.
4. **Environment reproducibility:** Python/PyTorch versions and frontend Node dependencies must be compatible; record actual versions used for testing.
5. **Evaluation provenance:** Dice/IoU require ground-truth masks and a defined split; classification metrics require labelled data. Store metric values with dataset split and model version rather than hard-coding UI numbers.
6. **Performance:** current API uses CPU inference. Measure actual latency on the demo machine and state whether the number is mock or real inference.

## Questions for supervisor / algorithm teammate

1. When will the trained segmentation checkpoint and inference/preprocessing code be available, and what exact output format will it return?
2. Should the agreed API response use base64 data URLs (current frontend expectation) or static/file URLs for masks and overlays?
3. Which evaluation split and metric artifact format should the backend expose for Dice, IoU, Macro-F1, and Accuracy?

## Evidence to bring to review

- Screenshot of `/docs` showing all endpoints.
- Terminal output for each executed smoke test, including expected failures.
- Frontend screenshot showing the mock notice.
- Completed test matrix with actual pass/fail and environment details.
- The three unresolved questions above.
