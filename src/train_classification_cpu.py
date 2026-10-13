import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

from classification_dataset import SkinLesionClassificationDataset
from classification_model import SkinLesionClassifier


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, "datasets", "ISIC2018", "classification")
IMAGE_DIR = os.path.join(DATA_ROOT, "images", "ISIC2018_Task3_Training_Input")
SPLIT_DIR = os.path.join(DATA_ROOT, "splits")
LABEL_CSV = os.path.join(SPLIT_DIR, "classification_labels.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "classification")
MODEL_PATH = os.path.join(OUTPUT_DIR, "classification_rgb_best_macro_f1.pth")

SEED = 42
BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 1e-3
CLASS_NAMES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cpu")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_dataset = SkinLesionClassificationDataset(
        IMAGE_DIR, LABEL_CSV, os.path.join(SPLIT_DIR, "train_cls.txt")
    )
    val_dataset = SkinLesionClassificationDataset(
        IMAGE_DIR, LABEL_CSV, os.path.join(SPLIT_DIR, "val_cls.txt")
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = SkinLesionClassifier(num_classes=len(CLASS_NAMES)).to(device)

    # Weights follow the class order above and help address the imbalanced training set.
    class_weights = torch.tensor(
        [1.2855, 0.2134, 2.7817, 4.3731, 1.3022, 12.3633, 10.1154],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-6
    )

    best_macro_f1 = -1.0
    print("RGB-only skin lesion classification training")
    print("Device:", device)
    print("Train:", len(train_dataset), "| Validation:", len(val_dataset))
    print("Masks required: no")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss_sum, train_labels, train_preds = 0.0, [], []
        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
            train_labels.extend(labels.numpy().tolist())
            train_preds.extend(logits.argmax(1).detach().numpy().tolist())

        model.eval()
        val_loss_sum, val_labels, val_preds = 0.0, [], []
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                logits = model(images)
                val_loss_sum += criterion(logits, labels).item()
                val_labels.extend(labels.numpy().tolist())
                val_preds.extend(logits.argmax(1).numpy().tolist())

        train_loss = train_loss_sum / max(1, len(train_loader))
        val_loss = val_loss_sum / max(1, len(val_loader))
        train_acc = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds, average="macro", zero_division=0)
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)
        scheduler.step(val_f1)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train loss {train_loss:.4f}, acc {train_acc:.4f}, macro-F1 {train_f1:.4f} | "
            f"val loss {val_loss:.4f}, acc {val_acc:.4f}, macro-F1 {val_f1:.4f}"
        )

        if val_f1 > best_macro_f1:
            best_macro_f1 = val_f1
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_accuracy": val_acc,
                    "val_macro_f1": val_f1,
                    "class_names": CLASS_NAMES,
                    "input_channels": 3,
                    "uses_segmentation_mask": False,
                },
                MODEL_PATH,
            )
            print("  Saved best RGB-only model:", MODEL_PATH)

    print("Training finished. Best validation Macro-F1:", f"{best_macro_f1:.4f}")


if __name__ == "__main__":
    main()
