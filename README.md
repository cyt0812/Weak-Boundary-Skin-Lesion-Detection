# Weak Boundary Guided Skin Lesion Detection System

A research project for skin lesion image analysis using deep learning. The current implementation provides a seven-class RGB skin lesion classification baseline, a FastAPI backend, and a React web interface.

The long-term goal is to investigate weak-boundary-guided skin lesion segmentation and its potential integration with disease classification.

> **Current status:** The web application and seven-class classification pipeline have been tested locally. Weak-boundary-guided segmentation and segmentation-classification fusion are still under development.

## 1. Project Overview

Skin lesion analysis can involve both lesion segmentation and disease classification. Segmentation identifies the lesion region, while classification predicts the lesion category from an image.

This project is being developed in stages:

- **Classification baseline:** Classifies an RGB skin lesion image into one of seven ISIC 2018 categories.
- **Backend API:** Uses FastAPI to load the trained classification checkpoint and return prediction results.
- **Web interface:** Uses React and Vite to upload an image and display the predicted class and class scores.
- **Future development:** Integrates lesion segmentation, investigates weak-boundary guidance, and evaluates possible segmentation-classification fusion strategies.

The current classification model is an RGB-only CNN. It does not use segmentation masks or segmentation features during inference.

## 2. Current Features

- Seven-class skin lesion image classification.
- CPU-compatible model inference.
- Image upload through a React web interface.
- FastAPI health-check and prediction endpoints.
- Display of the predicted class and scores for all seven classes.
- Training and evaluation scripts for the classification baseline.
- Local storage of datasets, model checkpoints, and training outputs.

## 3. Classification Categories

The model uses the following class order:

| Code | Description |
|---|---|
| MEL | Melanoma |
| NV | Melanocytic nevus |
| BCC | Basal cell carcinoma |
| AKIEC | Actinic keratoses and intraepithelial carcinoma / Bowen's disease |
| BKL | Benign keratosis-like lesions |
| DF | Dermatofibroma |
| VASC | Vascular lesions |

These categories follow the ISIC 2018 Task 3 classification labels.

## 4. Technology Stack

- **Programming language:** Python
- **Deep learning:** PyTorch
- **Image processing:** Pillow and NumPy
- **Data handling and evaluation:** pandas and scikit-learn
- **Backend:** FastAPI and Uvicorn
- **Frontend:** React and Vite
- **Development environment:** Conda
- **Version control:** Git and GitHub

The current application uses CPU inference. GPU acceleration is not required to run the classification API.

## 5. Project Structure

```text
Skin-Lesion-Detection/
├── backend/
│   └── main.py
├── frontend/
│   ├── package.json
│   ├── package-lock.json
│   └── src/
├── src/
│   ├── classification_dataset.py
│   ├── classification_model.py
│   ├── train_classification_cpu.py
│   ├── evaluate_classification_cpu.py
│   ├── evaluate_classification_test.py
│   ├── generate_classification_labels.py
│   ├── generate_classification_split.py
│   ├── predict_classification.py
│   └── test_classification_batch.py
├── datasets/
│   └── ISIC2018/
│       └── classification/
├── outputs/
│   └── classification/
│       └── classification_rgb_best_macro_f1.pth
├── requirements.txt
├── .gitignore
└── README.md
```

The exact contents of some directories may change as development continues. Datasets, trained checkpoints, and generated training outputs are stored locally and are excluded from Git.

## 6. Requirements

The project was developed and tested in a Windows environment with Python 3.10 and Node.js.

Required tools:

- Git
- Miniconda or Anaconda
- Python 3.10
- Node.js and npm

Python dependencies are listed in `requirements.txt`. Frontend dependencies are managed through `frontend/package.json` and `frontend/package-lock.json`.

## 7. Setup

### 7.1 Open the project directory

Open Anaconda Prompt or Command Prompt and navigate to the project root:

```bat
cd /d D:\GraduationProject\Skin-Lesion-Detection
```

Activate the existing Conda environment:

```bat
conda activate skin-lesion
```

Check Python:

```bat
python --version
```

### 7.2 Install Python dependencies

Run the following command from the project root:

```bat
python -m pip install -r requirements.txt
```

Verify the installation:

```bat
python -m pip check
```

The command should report that there are no broken requirements.

**Note:** PyTorch installation packages can differ between CPU and CUDA builds. If setting up a new environment, install a compatible PyTorch build for the intended hardware and verify it before training or inference.

### 7.3 Install frontend dependencies

Open a second terminal and navigate to the frontend directory:

```bat
cd /d D:\GraduationProject\Skin-Lesion-Detection\frontend
```

Install the locked frontend dependencies:

```bat
npm ci
```

Check the installed packages:

```bat
npm ls --depth=0
```

## 8. Dataset Preparation

The classification pipeline uses the ISIC 2018 Task 3 dataset.

Prepare the local dataset using the following layout:

```text
datasets/ISIC2018/classification/
├── images/
│   └── ISIC2018_Task3_Training_Input/
├── ground_truth/
│   └── ISIC2018_Task3_Training_GroundTruth/
│       └── ISIC2018_Task3_Training_GroundTruth.csv
└── splits/
    ├── classification_labels.csv
    ├── train_cls.txt
    ├── val_cls.txt
    └── test_cls.txt
```

The image directory should contain the corresponding training images, and the ground-truth CSV should contain the classification labels.

If the label table has not been generated, run:

```bat
python src\generate_classification_labels.py
```

If the split files do not exist, generate them with:

```bat
python src\generate_classification_split.py
```

Keep existing split files unchanged when reproducing previous experiments. If the splits are regenerated, document the change when comparing results.

The dataset is not included in this repository. Obtain it from the official dataset source and follow its terms of use.

## 9. Train the Classification Model

Activate the `skin-lesion` environment and run the command from the project root:

```bat
python src\train_classification_cpu.py
```

The current training configuration includes:

- Input size: 256 × 256 pixels
- Input channels: 3 (RGB)
- Batch size: 8
- Maximum epochs: 20
- Optimizer: Adam
- Initial learning rate: 0.001
- Random seed: 42
- Loss function: weighted cross-entropy
- Model selection: validation Macro-F1

The best checkpoint is saved to:

```text
outputs/classification/classification_rgb_best_macro_f1.pth
```

Training logs are saved locally in the same output directory.

These are the settings defined in the current training script; they may change as the project evolves.

## 10. Evaluate the Classification Model

Run the evaluation script from the project root:

```bat
python src\evaluate_classification_cpu.py
```

Review the script output to confirm which split is evaluated and where the metrics are saved.

Useful evaluation metrics include:

- Accuracy
- Macro Precision
- Macro Recall
- Macro-F1
- Per-class precision, recall, and F1-score
- Confusion matrix

Because the dataset has imbalanced classes, Accuracy should be considered alongside Macro-F1 and per-class results.

For fair evaluation, do not use test-set results to select model checkpoints or tune training parameters.

## 11. Run the Application

The application uses separate frontend and backend development servers. Start both servers and keep their terminal windows open.

### 11.1 Start the backend

In the first terminal, navigate to the project root and activate the Conda environment:

```bat
cd /d D:\GraduationProject\Skin-Lesion-Detection
conda activate skin-lesion
```

Make sure the trained checkpoint exists at:

```text
outputs/classification/classification_rgb_best_macro_f1.pth
```

Start FastAPI:

```bat
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

The backend API will be available at:

- API base URL: `http://127.0.0.1:8000`
- Interactive API documentation: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/api/health`

Open the health-check URL in a browser. The response should indicate that the service is running and show whether the model has loaded.

### 11.2 Start the frontend

In a second terminal:

```bat
cd /d D:\GraduationProject\Skin-Lesion-Detection\frontend
npm run dev
```

Vite normally serves the development application at:

```text
http://localhost:5173/
```

Open this URL in a browser, upload a supported skin lesion image, and review the prediction results.

The frontend development server and backend API must both be running for image prediction to work.

### 11.3 Test the prediction API

Open:

```text
http://127.0.0.1:8000/docs
```

Find the `POST /api/predict` endpoint, select an image file, and execute the request.

The endpoint returns the predicted class, class scores, model epoch, input size, and a research-use notice.

The current API accepts image uploads up to 10 MB.

## 12. Model and Data Management

The following files are intentionally excluded from version control:

- Local datasets
- Trained model checkpoints
- Generated training outputs
- Python virtual environments
- Frontend dependencies and build artifacts
- Local environment configuration files

In particular, `outputs/` is ignored by Git. Therefore, cloning the repository does not automatically provide the trained checkpoint required by the backend.

To run inference on another machine, prepare the dataset and train the model or obtain the corresponding checkpoint through an appropriate, authorized distribution method.

Do not commit large datasets, private files, or model weights accidentally.

## 13. Current Limitations and Future Work

The current implementation is a classification prototype, not a complete weak-boundary-guided lesion detection system.

The following work remains:

1. Establish and evaluate a lesion segmentation baseline.
2. Investigate weak-boundary information and segmentation quality.
3. Integrate segmentation inference into the backend.
4. Explore whether segmentation outputs or features can improve classification.
5. Evaluate the integrated system using suitable segmentation and classification metrics.
6. Improve error handling, interface usability, and reproducibility.
7. Document experiments and compare the proposed approach with appropriate baselines.

Any future segmentation or fusion results should be reported only after the corresponding implementation and evaluation have been completed.

## 14. Medical Disclaimer

This project is intended for academic research and software development only. It is not a medical device and must not be used to diagnose skin conditions or guide treatment.

The model's predicted class and scores are experimental outputs. The scores are not calibrated clinical probabilities and should not be interpreted as a measure of medical certainty.

## 15. License and Dataset Attribution

Check the licenses of the source code, pretrained models, and datasets before redistributing them.

If no project license has been selected, do not assume that the repository is licensed for unrestricted reuse. Add an appropriate license and dataset attribution once the applicable requirements have been confirmed.
