import os
import sys
import torch
import numpy as np

from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

# ============================================================
# Project path
# ============================================================

project_root = r"D:\GraduationProject\Skin-Lesion-Detection"

src_dir = os.path.join(
    project_root,
    "src"
)

sys.path.insert(
    0,
    src_dir
)

# ============================================================
# Import project modules
# ============================================================

from classification_dataset import SkinLesionClassificationDataset
from classification_model_fusion import SkinLesionFeatureFusionClassifier


# ============================================================
# Device
# ============================================================

device = torch.device("cpu")


# ============================================================
# Paths
# ============================================================

image_dir = os.path.join(
    project_root,
    "datasets",
    "ISIC2018",
    "classification",
    "images",
    "ISIC2018_Task3_Training_Input"
)

mask_dir = os.path.join(
    project_root,
    "outputs",
    "classification_coarse_masks"
)

label_csv = os.path.join(
    project_root,
    "datasets",
    "ISIC2018",
    "classification",
    "splits",
    "classification_labels.csv"
)

test_split = os.path.join(
    project_root,
    "datasets",
    "ISIC2018",
    "classification",
    "splits",
    "test_cls.txt"
)

model_path = os.path.join(
    project_root,
    "outputs",
    "classification",
    "classification_best_macro_f1.pth"
)


# ============================================================
# Class names
# ============================================================

class_names = [
    "MEL",
    "NV",
    "BCC",
    "AKIEC",
    "BKL",
    "DF",
    "VASC"
]


# ============================================================
# Header
# ============================================================

print("==============================================")
print("Skin Lesion Classification Test Evaluation")
print("==============================================")

print("Device:", device)

print()
print("Model:")
print(model_path)


# ============================================================
# Check model
# ============================================================

if not os.path.exists(model_path):

    raise FileNotFoundError(
        f"找不到模型:\n{model_path}"
    )


# ============================================================
# Load Test Dataset
# ============================================================

print()
print("========== Loading Test Dataset ==========")

test_dataset = SkinLesionClassificationDataset(
    image_dir=image_dir,
    mask_dir=mask_dir,
    label_csv=label_csv,
    split_file=test_split
)

test_loader = DataLoader(
    test_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0
)

print()
print("Test dataset:", len(test_dataset))
print("Test batches:", len(test_loader))


# ============================================================
# Load Model
# ============================================================

print()
print("========== Loading Model ==========")

model = SkinLesionFeatureFusionClassifier(
    num_classes=7
)

checkpoint = torch.load(
    model_path,
    map_location=device
)

if "model_state_dict" in checkpoint:

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print("Model loaded successfully.")

    if "epoch" in checkpoint:

        print(
            "Saved epoch:",
            checkpoint["epoch"]
        )

    if "val_macro_f1" in checkpoint:

        print(
            "Saved validation Macro F1:",
            checkpoint["val_macro_f1"]
        )

else:

    model.load_state_dict(
        checkpoint
    )

    print("Model loaded successfully.")


model = model.to(device)

model.eval()


# ============================================================
# Evaluation
# ============================================================

print()
print("========== Testing ==========")

all_labels = []
all_predictions = []

with torch.no_grad():

    for batch_index, batch in enumerate(
        test_loader,
        start=1
    ):

        image = batch["image"].to(device)

        mask = batch["mask"].to(device)

        labels = batch["label"].to(device)

        logits = model(image, mask)

        predictions = torch.argmax(
            logits,
            dim=1
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        if batch_index % 100 == 0:

            print(
                f"Batch [{batch_index}/{len(test_loader)}]"
            )


# ============================================================
# Convert to NumPy
# ============================================================

all_labels = np.array(
    all_labels
)

all_predictions = np.array(
    all_predictions
)


# ============================================================
# Overall Metrics
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision, recall, f1, support = (
    precision_recall_fscore_support(
        all_labels,
        all_predictions,
        labels=np.arange(7),
        zero_division=0
    )
)

macro_precision = precision.mean()

macro_recall = recall.mean()

macro_f1 = f1.mean()


# ============================================================
# Overall Results
# ============================================================

print()
print("==============================================")
print("Final Test Results")
print("==============================================")

print(
    f"Accuracy:       {accuracy:.4f}"
)

print(
    f"Macro Precision: {macro_precision:.4f}"
)

print(
    f"Macro Recall:    {macro_recall:.4f}"
)

print(
    f"Macro F1:        {macro_f1:.4f}"
)


# ============================================================
# Per-Class Results
# ============================================================

print()
print("==============================================")
print("Per-Class Test Results")
print("==============================================")

print(
    f"{'Class':<10}"
    f"{'Precision':>12}"
    f"{'Recall':>12}"
    f"{'F1':>12}"
    f"{'Support':>12}"
)

for i, name in enumerate(class_names):

    print(
        f"{name:<10}"
        f"{precision[i]:>12.4f}"
        f"{recall[i]:>12.4f}"
        f"{f1[i]:>12.4f}"
        f"{support[i]:>12}"
    )


# ============================================================
# Classification Report
# ============================================================

print()
print("==============================================")
print("Classification Report")
print("==============================================")

print(
    classification_report(
        all_labels,
        all_predictions,
        labels=np.arange(7),
        target_names=class_names,
        digits=4,
        zero_division=0
    )
)


# ============================================================
# Confusion Matrix
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=np.arange(7)
)

print()
print("==============================================")
print("Confusion Matrix")
print("==============================================")

print(
    "Rows = True Label"
)

print(
    "Columns = Predicted Label"
)

print()

print(
    f"{'':>10}"
    + "".join(
        f"{name:>8}"
        for name in class_names
    )
)

for i, name in enumerate(class_names):

    print(
        f"{name:>10}"
        + "".join(
            f"{cm[i, j]:>8}"
            for j in range(7)
        )
    )


# ============================================================
# Class Distribution
# ============================================================

print()
print("==============================================")
print("Class Distribution")
print("==============================================")

print(
    f"{'Class':<10}"
    f"{'True':>12}"
    f"{'Predicted':>12}"
)

for i, name in enumerate(class_names):

    true_count = int(
        np.sum(all_labels == i)
    )

    predicted_count = int(
        np.sum(all_predictions == i)
    )

    print(
        f"{name:<10}"
        f"{true_count:>12}"
        f"{predicted_count:>12}"
    )


# ============================================================
# Finished
# ============================================================

print()
print("==============================================")
print("Test Evaluation Finished")
print("==============================================")