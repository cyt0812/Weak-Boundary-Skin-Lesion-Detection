import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

from classification_dataset import SkinLesionClassificationDataset
from classification_model import SkinLesionClassifier


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, "datasets", "ISIC2018", "classification")
IMAGE_DIR = os.path.join(DATA_ROOT, "images", "ISIC2018_Task3_Training_Input")
SPLIT_DIR = os.path.join(DATA_ROOT, "splits")
LABEL_CSV = os.path.join(SPLIT_DIR, "classification_labels.csv")
TEST_SPLIT = os.path.join(SPLIT_DIR, "test_cls.txt")
MODEL_PATH = os.path.join(PROJECT_ROOT, "outputs", "classification", "classification_rgb_best_macro_f1.pth")
CLASS_NAMES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]


def main():
    device = torch.device("cpu")
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"找不到 RGB-only 分类模型：{MODEL_PATH}\n"
            "请先运行：python src\\train_classification_cpu.py"
        )

    dataset = SkinLesionClassificationDataset(IMAGE_DIR, LABEL_CSV, TEST_SPLIT)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model = SkinLesionClassifier(num_classes=7)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    labels_all, preds_all = [], []
    with torch.no_grad():
        for i, batch in enumerate(loader, start=1):
            logits = model(batch["image"].to(device))
            labels_all.extend(batch["label"].numpy().tolist())
            preds_all.extend(logits.argmax(1).numpy().tolist())
            if i % 100 == 0:
                print(f"Batch [{i}/{len(loader)}]")

    y_true = np.asarray(labels_all)
    y_pred = np.asarray(preds_all)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(7), zero_division=0
    )
    print("\n========== RGB-only Test Results ==========")
    print(f"Test images: {len(dataset)}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Macro Precision: {precision.mean():.4f}")
    print(f"Macro Recall: {recall.mean():.4f}")
    print(f"Macro F1: {f1.mean():.4f}")
    print("\nPer-class results:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"{name:6s} precision={precision[i]:.4f} recall={recall[i]:.4f} "
              f"F1={f1[i]:.4f} support={support[i]}")
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, labels=np.arange(7),
                                target_names=CLASS_NAMES, zero_division=0))
    print("Confusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(y_true, y_pred, labels=np.arange(7)))


if __name__ == "__main__":
    main()
