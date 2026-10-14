import os
import torch
from torch.utils.data import DataLoader

from classification_dataset import SkinLesionClassificationDataset
from classification_model import SkinLesionClassifier


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_ROOT = os.path.join(
    PROJECT_ROOT, "datasets", "ISIC2018", "classification"
)

IMAGE_DIR = os.path.join(
    DATA_ROOT, "images", "ISIC2018_Task3_Training_Input"
)

SPLIT_DIR = os.path.join(DATA_ROOT, "splits")

LABEL_CSV = os.path.join(
    SPLIT_DIR, "classification_labels.csv"
)

TRAIN_SPLIT = os.path.join(SPLIT_DIR, "train_cls.txt")

BATCH_SIZE = 8


def main():
    torch.manual_seed(42)

    dataset = SkinLesionClassificationDataset(
        image_dir=IMAGE_DIR,
        label_csv=LABEL_CSV,
        split_file=TRAIN_SPLIT,
        image_size=(256, 256),
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    batch = next(iter(loader))

    images = batch["image"]
    labels = batch["label"]

    print("\n--- Batch data check ---")
    print("Image tensor shape:", tuple(images.shape))
    print("Label tensor shape:", tuple(labels.shape))
    print("Pixel minimum:", images.min().item())
    print("Pixel maximum:", images.max().item())
    print("Image tensor dtype:", images.dtype)
    print("Label tensor dtype:", labels.dtype)

    assert tuple(images.shape) == (BATCH_SIZE, 3, 256, 256)
    assert tuple(labels.shape) == (BATCH_SIZE,)
    assert images.min().item() >= 0.0
    assert images.max().item() <= 1.0

    model = SkinLesionClassifier(num_classes=7)
    model.eval()

    with torch.no_grad():
        logits = model(images)

    print("\n--- Model forward-pass check ---")
    print("Output logits shape:", tuple(logits.shape))
    print("Number of classes:", logits.shape[1])

    assert tuple(logits.shape) == (BATCH_SIZE, 7)

    predictions = logits.argmax(dim=1)

    print("Predicted class indices:", predictions.tolist())
    print("\nAll batch checks passed.")


if __name__ == "__main__":
    main()