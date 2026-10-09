import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import (
accuracy_score,
f1_score
)

# ============================================================

# 1. Project paths

# ============================================================

PROJECT_ROOT = r"D:\GraduationProject\Skin-Lesion-Detection"

SRC_DIR = os.path.join(
PROJECT_ROOT,
"src"
)

sys.path.insert(0, SRC_DIR)

# ============================================================

# 2. Import project code

# ============================================================

from classification_dataset import SkinLesionClassificationDataset
from classification_model_fusion import SkinLesionFeatureFusionClassifier

# ============================================================

# 3. Device

# ============================================================

device = torch.device("cpu")

print("==============================================")
print("Skin Lesion Classification Training")
print("==============================================")
print("Device:", device)

# ============================================================

# 4. Dataset paths

# ============================================================

image_dir = os.path.join(
PROJECT_ROOT,
"datasets",
"ISIC2018",
"classification",
"images",
"ISIC2018_Task3_Training_Input"
)

mask_dir = os.path.join(
PROJECT_ROOT,
"outputs",
"classification_coarse_masks"
)

label_csv = os.path.join(
PROJECT_ROOT,
"datasets",
"ISIC2018",
"classification",
"splits",
"classification_labels.csv"
)

split_dir = os.path.join(
PROJECT_ROOT,
"datasets",
"ISIC2018",
"classification",
"splits"
)

train_split = os.path.join(
split_dir,
"train_cls.txt"
)

val_split = os.path.join(
split_dir,
"val_cls.txt"
)

# ============================================================

# 5. Create Dataset

# ============================================================

print()
print("========== Loading Dataset ==========")

train_dataset = SkinLesionClassificationDataset(
image_dir=image_dir,
mask_dir=mask_dir,
label_csv=label_csv,
split_file=train_split
)

val_dataset = SkinLesionClassificationDataset(
image_dir=image_dir,
mask_dir=mask_dir,
label_csv=label_csv,
split_file=val_split
)

print()
print("Train dataset:", len(train_dataset))
print("Val dataset:", len(val_dataset))

# ============================================================

# 6. DataLoader

# ============================================================

train_loader = DataLoader(
train_dataset,
batch_size=8,
shuffle=True,
num_workers=0
)

val_loader = DataLoader(
val_dataset,
batch_size=4,
shuffle=False,
num_workers=0
)

print()
print("Train batches:", len(train_loader))
print("Val batches:", len(val_loader))

# ============================================================

# 7. Create Model

# ============================================================

print()
print("========== Creating Model ==========")

model = SkinLesionFeatureFusionClassifier(
num_classes=7
)

model = model.to(device)

total_params = sum(
p.numel()
for p in model.parameters()
)

print("Model parameters:", total_params)

# ============================================================

# 8. Class Weights

# ============================================================

class_weights = torch.tensor(
[
1.2855,     # MEL
0.2134,     # NV
2.7817,     # BCC
4.3731,     # AKIEC
1.3022,     # BKL
12.3633,    # DF
10.1154     # VASC
],
dtype=torch.float32
).to(device)

print()
print("Class weights:")
print(class_weights)

# ============================================================

# 9. Loss Function

# ============================================================

criterion = nn.CrossEntropyLoss(
weight=class_weights
)

# ============================================================

# 10. Optimizer

# ============================================================

optimizer = torch.optim.Adam(
model.parameters(),
lr=1e-3
)

# ============================================================

# 11. Learning Rate Scheduler

# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
optimizer,
mode="max",
factor=0.5,
patience=2,
min_lr=1e-6
)

# ============================================================

# 12. Training parameters

# ============================================================

num_epochs = 20

best_val_macro_f1 = 0.0

best_epoch = 0

# ============================================================

# 13. Output directory

# ============================================================

output_dir = os.path.join(
PROJECT_ROOT,
"outputs",
"classification"
)

os.makedirs(
output_dir,
exist_ok=True
)

best_model_path = os.path.join(
output_dir,
"classification_best_macro_f1.pth"
)

# ============================================================

# 14. Training

# ============================================================

for epoch in range(num_epochs):

    print()
    print("==============================================")
    print(
        f"Epoch {epoch + 1}/{num_epochs}"
    )
    print("==============================================")

# --------------------------------------------------------
# Training mode
# --------------------------------------------------------

    model.train()

    running_loss = 0.0

    train_correct = 0
    train_total = 0

    train_labels = []
    train_predictions = []

    for batch_idx, batch in enumerate(train_loader):

        image = batch["image"].to(device)

        mask = batch["mask"].to(device)

        labels = batch["label"].to(device)

        optimizer.zero_grad()

        logits = model(image, mask)

        loss = criterion(
            logits,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        predictions = torch.argmax(
            logits,
            dim=1
        )

        train_correct += (
            predictions == labels
        ).sum().item()

        train_total += labels.size(0)

        train_labels.extend(
            labels.cpu().numpy()
        )

        train_predictions.extend(
            predictions.cpu().numpy()
        )

        if (batch_idx + 1) % 200 == 0:

            current_loss = (
                running_loss /
                (batch_idx + 1)
            )

            current_accuracy = (
                train_correct /
                train_total
            )

            print(
                f"Batch [{batch_idx + 1}/{len(train_loader)}] "
                f"Loss: {current_loss:.4f} "
                f"Accuracy: {current_accuracy:.4f}"
            )


    # --------------------------------------------------------
    # Training statistics
    # --------------------------------------------------------

    train_loss = (
        running_loss /
        len(train_loader)
    )

    train_accuracy = accuracy_score(
        train_labels,
        train_predictions
    )

    train_macro_f1 = f1_score(
        train_labels,
        train_predictions,
        average="macro",
        zero_division=0
    )


    # ========================================================
    # Validation
    # ========================================================

    model.eval()

    val_running_loss = 0.0

    val_labels = []
    val_predictions = []

    with torch.no_grad():

        for batch_idx, batch in enumerate(val_loader):

            image = batch["image"].to(device)

            mask = batch["mask"].to(device)

            labels = batch["label"].to(device)

            logits = model(image, mask)

            loss = criterion(
                logits,
                labels
            )

            val_running_loss += loss.item()

            predictions = torch.argmax(
                logits,
                dim=1
            )

            val_labels.extend(
                labels.cpu().numpy()
            )

            val_predictions.extend(
                predictions.cpu().numpy()
            )

            if (batch_idx + 1) % 50 == 0:

                print(
                    f"Validation [{batch_idx + 1}/{len(val_loader)}]"
                )


    # --------------------------------------------------------
    # Validation statistics
    # --------------------------------------------------------

    val_loss = (
        val_running_loss /
        len(val_loader)
    )

    val_accuracy = accuracy_score(
        val_labels,
        val_predictions
    )

    val_macro_f1 = f1_score(
        val_labels,
        val_predictions,
        average="macro",
        zero_division=0
    )


    # ========================================================
    # Scheduler
    # ========================================================

    scheduler.step(
        val_macro_f1
    )

    current_lr = optimizer.param_groups[0]["lr"]


    # ========================================================
    # Print Epoch Results
    # ========================================================

    print()
    print("----------------------------------------------")
    print("Epoch Results")
    print("----------------------------------------------")

    print(
        f"Train Loss:       {train_loss:.4f}"
    )

    print(
        f"Train Accuracy:   {train_accuracy:.4f}"
    )

    print(
        f"Train Macro F1:   {train_macro_f1:.4f}"
    )

    print(
        f"Val Loss:         {val_loss:.4f}"
    )

    print(
        f"Val Accuracy:     {val_accuracy:.4f}"
    )

    print(
        f"Val Macro F1:     {val_macro_f1:.4f}"
    )

    print(
        f"Learning Rate:    {current_lr:.7f}"
    )


    # ========================================================
    # Save Best Model
    # ========================================================

    if val_macro_f1 > best_val_macro_f1:

        best_val_macro_f1 = val_macro_f1

        best_epoch = epoch + 1

        torch.save(
            {
                "epoch": epoch + 1,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "train_loss":
                    train_loss,

                "train_accuracy":
                    train_accuracy,

                "train_macro_f1":
                    train_macro_f1,

                "val_loss":
                    val_loss,

                "val_accuracy":
                    val_accuracy,

                "val_macro_f1":
                    val_macro_f1,

                "class_weights":
                    class_weights.cpu(),

                "learning_rate":
                    current_lr
            },
            best_model_path
        )

        print()
        print(">>> New best model saved!")

        print(
            "Best Val Macro F1:",
            f"{best_val_macro_f1:.4f}"
        )

        print(
            "Saved to:",
            best_model_path
        )

# ============================================================

# 15. Training Finished

# ============================================================

print()
print("==============================================")
print("Training Finished")
print("==============================================")

print(
"Best Epoch:",
best_epoch
)

print(
"Best Val Macro F1:",
f"{best_val_macro_f1:.4f}"
)

print()
print("Best model:")
print(best_model_path)

print()
print("==============================================")
