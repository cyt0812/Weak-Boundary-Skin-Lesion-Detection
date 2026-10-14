import os
import random
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

from classification_dataset import SkinLesionClassificationDataset
from classification_model import SkinLesionClassifier


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(
    PROJECT_ROOT, "datasets", "ISIC2018", "classification"
)
IMAGE_DIR = os.path.join(
    DATA_ROOT, "images", "ISIC2018_Task3_Training_Input"
)
SPLIT_DIR = os.path.join(DATA_ROOT, "splits")
LABEL_CSV = os.path.join(SPLIT_DIR, "classification_labels.csv")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "classification")
MODEL_PATH = os.path.join(
    OUTPUT_DIR, "classification_rgb_best_macro_f1.pth"
)
LOG_PATH = os.path.join(
    OUTPUT_DIR, "classification_training_log.csv"
)

SEED = 42
BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 1e-3

CLASS_NAMES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
NUM_CLASSES = len(CLASS_NAMES)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cpu")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_dataset = SkinLesionClassificationDataset(
        IMAGE_DIR,
        LABEL_CSV,
        os.path.join(SPLIT_DIR, "train_cls.txt"),
        image_size=(256, 256),
    )

    val_dataset = SkinLesionClassificationDataset(
        IMAGE_DIR,
        LABEL_CSV,
        os.path.join(SPLIT_DIR, "val_cls.txt"),
        image_size=(256, 256),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # Calculate class weights using the training split only.
    train_labels_for_weights = np.array(
        [
            train_dataset.labels[name]
            for name in train_dataset.image_names
        ],
        dtype=np.int64,
    )

    class_weights_np = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(NUM_CLASSES),
        y=train_labels_for_weights,
    )

    class_weights = torch.tensor(
        class_weights_np,
        dtype=torch.float32,
        device=device,
    )

    print("\nClass weights calculated from training split only:")
    for name, weight in zip(CLASS_NAMES, class_weights_np):
        print(f"  {name}: {weight:.4f}")

    model = SkinLesionClassifier(num_classes=NUM_CLASSES).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )

    best_macro_f1 = -1.0

    # Write a fresh log for this training run.
    with open(LOG_PATH, "w", newline="", encoding="utf-8-sig") as log_file:
        writer = csv.writer(log_file)
        writer.writerow([
            "epoch",
            "learning_rate",
            "train_loss",
            "train_accuracy",
            "train_macro_f1",
            "val_loss",
            "val_accuracy",
            "val_macro_f1",
        ])

    print("\nRGB-only skin lesion classification baseline")
    print("Device:", device)
    print("Input size: 256 x 256")
    print("Input: RGB images only")
    print("Pixel normalization: divide by 255, range [0, 1]")
    print("Train images:", len(train_dataset))
    print("Validation images:", len(val_dataset))
    print("Batch size:", BATCH_SIZE)
    print("Epochs:", EPOCHS)
    print("Segmentation masks: not used")
    print("Test split: not used during training")
    print()

    for epoch in range(1, EPOCHS + 1):
        model.train()

        train_loss_sum = 0.0
        train_labels = []
        train_preds = []

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
            train_preds.extend(
                logits.argmax(dim=1).detach().numpy().tolist()
            )

        model.eval()

        val_loss_sum = 0.0
        val_labels = []
        val_preds = []

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)

                logits = model(images)
                loss = criterion(logits, labels)

                val_loss_sum += loss.item()
                val_labels.extend(labels.numpy().tolist())
                val_preds.extend(
                    logits.argmax(dim=1).numpy().tolist()
                )

        train_loss = train_loss_sum / max(1, len(train_loader))
        val_loss = val_loss_sum / max(1, len(val_loader))

        train_acc = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(
            train_labels,
            train_preds,
            average="macro",
            zero_division=0,
        )

        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(
            val_labels,
            val_preds,
            average="macro",
            zero_division=0,
        )

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_f1)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"lr {current_lr:.6g} | "
            f"train loss {train_loss:.4f}, "
            f"acc {train_acc:.4f}, "
            f"macro-F1 {train_f1:.4f} | "
            f"val loss {val_loss:.4f}, "
            f"acc {val_acc:.4f}, "
            f"macro-F1 {val_f1:.4f}"
        )

        with open(
            LOG_PATH, "a", newline="", encoding="utf-8-sig"
        ) as log_file:
            writer = csv.writer(log_file)
            writer.writerow([
                epoch,
                current_lr,
                train_loss,
                train_acc,
                train_f1,
                val_loss,
                val_acc,
                val_f1,
            ])

        if val_f1 > best_macro_f1:
            best_macro_f1 = val_f1

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_accuracy": val_acc,
                    "val_macro_f1": val_f1,
                    "class_names": CLASS_NAMES,
                    "num_classes": NUM_CLASSES,
                    "input_size": (256, 256),
                    "input_channels": 3,
                    "normalization": "RGB float32 divided by 255, range [0, 1]",
                    "class_weights": class_weights_np.tolist(),
                    "seed": SEED,
                    "uses_segmentation_mask": False,
                },
                MODEL_PATH,
            )

            print("  Saved best model:", MODEL_PATH)

    print("\nTraining finished.")
    print(f"Best validation Macro-F1: {best_macro_f1:.4f}")
    print("Best model:", MODEL_PATH)
    print("Training log:", LOG_PATH)


if __name__ == "__main__":
    main()
