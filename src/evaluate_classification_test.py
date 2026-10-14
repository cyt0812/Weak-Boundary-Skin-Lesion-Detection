
import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
SRC_DIR = os.path.dirname(os.path.abspath(__file__))

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from classification_dataset import SkinLesionClassificationDataset
from classification_model import SkinLesionClassifier


# --------------------------------------------------
# 1. Paths and settings
# --------------------------------------------------
DATA_ROOT = os.path.join(
    PROJECT_ROOT, "datasets", "ISIC2018", "classification"
)

IMAGE_DIR = os.path.join(
    DATA_ROOT, "images", "ISIC2018_Task3_Training_Input"
)
SPLIT_DIR = os.path.join(DATA_ROOT, "splits")
LABEL_CSV = os.path.join(SPLIT_DIR, "classification_labels.csv")
TEST_SPLIT = os.path.join(SPLIT_DIR, "test_cls.txt")

BEST_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "classification",
    "classification_rgb_best_macro_f1.pth",
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT, "outputs", "classification", "test_evaluation"
)

CLASS_NAMES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
NUM_CLASSES = 7
BATCH_SIZE = 8
DEVICE = torch.device("cpu")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Confirm required files exist.
    for path in [IMAGE_DIR, LABEL_CSV, TEST_SPLIT, BEST_MODEL_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required path not found: {path}")

    print("=" * 60)
    print("ISIC 2018 RGB Classification: Independent Test Evaluation")
    print("=" * 60)
    print("Device:", DEVICE)
    print("Best checkpoint:", BEST_MODEL_PATH)

    # --------------------------------------------------
    # 2. Load the independent test split only
    # --------------------------------------------------
    test_dataset = SkinLesionClassificationDataset(
        image_dir=IMAGE_DIR,
        label_csv=LABEL_CSV,
        split_file=TEST_SPLIT,
        image_size=(256, 256),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print("Test images:", len(test_dataset))

    # --------------------------------------------------
    # 3. Load the saved best model
    # --------------------------------------------------
    model = SkinLesionClassifier(num_classes=NUM_CLASSES).to(DEVICE)

    # This checkpoint was created by this project's own training script.
    try:
        checkpoint = torch.load(
            BEST_MODEL_PATH,
            map_location=DEVICE,
            weights_only=False,
        )
    except TypeError:
        # Compatibility with older PyTorch versions.
        checkpoint = torch.load(
            BEST_MODEL_PATH,
            map_location=DEVICE,
        )

    if isinstance(checkpoint, dict):
        state_dict = None

        for key in (
            "model_state_dict",
            "model_state",
            "state_dict",
            "model",
        ):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                state_dict = checkpoint[key]
                break

        if state_dict is None:
            # Also support a checkpoint containing only model parameters.
            if checkpoint and all(
                isinstance(value, torch.Tensor)
                for value in checkpoint.values()
            ):
                state_dict = checkpoint

        if state_dict is None:
            raise KeyError(
                "Could not locate model weights in checkpoint. "
                "Please inspect the checkpoint-saving code."
            )
    else:
        raise TypeError("Unexpected checkpoint format.")

    model.load_state_dict(state_dict)
    model.eval()

    saved_epoch = checkpoint.get("epoch", "unknown") \
        if isinstance(checkpoint, dict) else "unknown"
    saved_val_f1 = checkpoint.get("val_macro_f1", "unknown") \
        if isinstance(checkpoint, dict) else "unknown"

    print("Checkpoint epoch:", saved_epoch)
    print("Checkpoint validation Macro-F1:", saved_val_f1)

    # --------------------------------------------------
    # 4. Predict test images
    # --------------------------------------------------
    all_labels = []
    all_predictions = []
    all_probabilities = []
    all_names = []

    with torch.no_grad():
        for batch_index, batch in enumerate(test_loader, start=1):
            images = batch["image"].to(DEVICE)
            labels = batch["label"]

            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_predictions.extend(predictions.cpu().numpy().tolist())
            all_probabilities.extend(probabilities.cpu().numpy().tolist())
            all_names.extend(batch["name"])

            if batch_index % 25 == 0 or batch_index == len(test_loader):
                print(
                    f"Evaluated batches: {batch_index}/{len(test_loader)}"
                )

    y_true = np.asarray(all_labels, dtype=int)
    y_pred = np.asarray(all_predictions, dtype=int)
    probabilities = np.asarray(all_probabilities)

    if len(y_true) != len(test_dataset):
        raise RuntimeError("Prediction count does not match test dataset size.")

    # --------------------------------------------------
    # 5. Calculate metrics
    # --------------------------------------------------
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        average="macro",
        zero_division=0,
    )

    report_text = classification_report(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
    )

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Test Macro-F1: {macro_f1:.4f}")
    print("\nPer-class classification report:")
    print(report_text)
    print("Confusion matrix (rows=true, columns=predicted):")
    print(cm)

    # --------------------------------------------------
    # 6. Save results
    # --------------------------------------------------
    metrics_path = os.path.join(OUTPUT_DIR, "test_metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("ISIC 2018 RGB Classification Test Results\n")
        f.write(f"Checkpoint epoch: {saved_epoch}\n")
        f.write(f"Test image count: {len(y_true)}\n")
        f.write(f"Test Accuracy: {accuracy:.6f}\n")
        f.write(f"Test Macro-F1: {macro_f1:.6f}\n\n")
        f.write("Per-class classification report:\n")
        f.write(report_text)
        f.write("\nConfusion matrix (rows=true, columns=predicted):\n")
        f.write(np.array2string(cm))

    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(
        os.path.join(OUTPUT_DIR, "classification_report.csv"),
        encoding="utf-8-sig",
        index=True,
    )

    cm_df = pd.DataFrame(
        cm,
        index=[f"True_{name}" for name in CLASS_NAMES],
        columns=[f"Pred_{name}" for name in CLASS_NAMES],
    )
    cm_df.to_csv(
        os.path.join(OUTPUT_DIR, "confusion_matrix.csv"),
        encoding="utf-8-sig",
    )

    prediction_data = {
        "image_name": all_names,
        "true_label": y_true,
        "true_class": [CLASS_NAMES[i] for i in y_true],
        "predicted_label": y_pred,
        "predicted_class": [CLASS_NAMES[i] for i in y_pred],
    }

    for class_index, class_name in enumerate(CLASS_NAMES):
        prediction_data[f"prob_{class_name}"] = probabilities[:, class_index]

    pd.DataFrame(prediction_data).to_csv(
        os.path.join(OUTPUT_DIR, "test_predictions.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    print("\nEvaluation completed.")
    print("Results saved to:", OUTPUT_DIR)
    print(" - test_metrics.txt")
    print(" - classification_report.csv")
    print(" - confusion_matrix.csv")
    print(" - test_predictions.csv")


if __name__ == "__main__":
    main()
