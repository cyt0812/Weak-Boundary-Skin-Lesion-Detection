# Weak Boundary Guided Skin Lesion Detection System

## 1. Project Overview

This project aims to develop a skin lesion detection system using deep learning. The current implementation includes a lesion segmentation model and a classification model.

The project is being developed as a graduation project.

## 2. Project Structure

```text
Skin-Lesion-Detection/
├── algorithms/       # Algorithm reference code
├── backend/          # Backend development
├── datasets/         # Datasets (not uploaded to GitHub)
├── docs/             # Project documentation
├── frontend/         # Frontend development
├── models/           # Model files
├── outputs/          # Training outputs and checkpoints
├── src/              # Python source code
└── README.md         # Project documentation
```

## 3. Main Components

* `src/predict.py`: Image prediction pipeline.
* `src/dataset.py`: Dataset processing.
* `src/classification_model.py`: Classification model.
* `src/classification_model_fusion.py`: Classification model with feature fusion.
* `src/train_coarse_cpu.py`: Coarse segmentation model training.
* `src/train_classification_cpu.py`: Classification model training.

## 4. Development Environment

The Python environment is managed using Conda.

* Python: 3.10
* Deep learning framework: PyTorch
* Development platform: Windows

The required packages and exact execution commands will be documented after the environment configuration is finalized.

## 5. Data and Model Files

Datasets, trained model weights, and training outputs are excluded from the Git repository to avoid uploading large files.

Team members should configure their local datasets and obtain the required model weights before running the complete pipeline.

## 6. Team Collaboration

The project source code is managed using Git and GitHub. Team members should coordinate code changes and test their changes before merging them into the main project.
