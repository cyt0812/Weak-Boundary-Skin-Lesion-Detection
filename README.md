# Skin Lesion Classification

A CPU-compatible, RGB-only seven-class skin-lesion classification project based on the ISIC 2018 Task 3 dataset. The main pipeline does **not** use a segmentation model, segmentation masks, or segmentation checkpoints.

## Classes

The class order used by the labels, training script, and evaluation script is:

1. MEL
2. NV
3. BCC
4. AKIEC
5. BKL
6. DF
7. VASC

## Project structure

```text
Skin-Lesion-Detection/
├── datasets/ISIC2018/classification/   # Local dataset; not committed
├── outputs/classification/             # Local checkpoints; not committed
├── src/
│   ├── classification_dataset.py       # RGB images + labels only
│   ├── classification_model.py         # Three-channel RGB classifier
│   ├── train_classification_cpu.py     # Train and select by validation Macro-F1
│   ├── evaluate_classification_cpu.py  # Test-set metrics and confusion matrix
│   ├── generate_classification_labels.py
│   └── generate_classification_split.py
├── .gitignore
└── README.md
```

## Environment

Use the existing Conda environment if it already has Python 3.10, PyTorch, pandas, Pillow, NumPy, and scikit-learn installed.

## Dataset layout

Place the ISIC 2018 Task 3 files locally as follows:

```text
datasets/ISIC2018/classification/
├── images/ISIC2018_Task3_Training_Input/*.jpg
├── ground_truth/ISIC2018_Task3_Training_GroundTruth/
│   └── ISIC2018_Task3_Training_GroundTruth.csv
└── splits/
    ├── classification_labels.csv
    ├── train_cls.txt
    ├── val_cls.txt
    └── test_cls.txt
```

The dataset and split files are local and must not be uploaded if licensing or size restrictions apply. The split generator uses a fixed random seed and stratification; do not regenerate splits after test results have been inspected unless you document the change.

## Run

Run commands from the project root in the activated Conda environment.

Generate the label table if needed:

```bat
python src\generate_classification_labels.py
```

Create train/validation/test split files if they do not already exist:

```bat
python src\generate_classification_split.py
```

Train the RGB-only classifier:

```bat
python src\train_classification_cpu.py
```

Evaluate the saved RGB-only model on the test split:

```bat
python src\evaluate_classification_cpu.py
```

The RGB-only model uses three input channels and does not require any mask directory. Existing four-channel RGB-plus-mask checkpoints are **not compatible** with this model and must not be presented as results from the RGB-only pipeline. Train a new checkpoint before evaluation.

## Metrics

The evaluation script reports Accuracy, Macro Precision, Macro Recall, Macro-F1, per-class scores, a classification report, and a confusion matrix. Because ISIC 2018 classes are imbalanced, report Macro-F1 and per-class results alongside Accuracy.

## Scope

This repository currently focuses on classification only. Segmentation training, segmentation inference, mask-generation scripts, and segmentation checkpoints are not part of the maintained classification pipeline.
