from pathlib import Path
import sys
import torch
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(r"D:\GraduationProject\Skin-Lesion-Detection")
MB_DCNN_ROOT = Path(r"D:\GraduationProject\MB-DCNN")

DEVICE = torch.device("cpu")
IMAGE_SIZE_SEG = (256, 256)
IMAGE_SIZE_CLS = (128, 128)

print("Predict module loaded successfully")
print("Device:", DEVICE)
print("Segmentation input size:", IMAGE_SIZE_SEG)
print("Classification input size:", IMAGE_SIZE_CLS)

COARSE_CHECKPOINT = PROJECT_ROOT / "outputs" / "coarse_sn" / "coarse_sn_epoch_4.pth"


def load_coarse_model():
    global COARSE_MODEL
    if COARSE_MODEL is not None:
        return COARSE_MODEL

    sys.path.insert(0, str(MB_DCNN_ROOT))
    from net.models import deeplabv3plus

    model = deeplabv3plus(num_classes=2)
    checkpoint = torch.load(COARSE_CHECKPOINT, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(DEVICE)
    model.eval()
    COARSE_MODEL = model

    print("CoarseSN loaded successfully")
    print("CoarseSN checkpoint epoch:", checkpoint["epoch"])
    print("CoarseSN validation Dice:", checkpoint["val_dice"])

    return model


CLASSIFICATION_CHECKPOINT = PROJECT_ROOT / "outputs" / "classification" / "classification_best_macro_f1.pth"


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class SkinLesionFeatureFusionClassifier(torch.nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()

        self.image_branch = torch.nn.Sequential(
            ConvBlock(3, 16),
            torch.nn.MaxPool2d(2, 2),
            ConvBlock(16, 32),
            torch.nn.MaxPool2d(2, 2),
            ConvBlock(32, 64),
            torch.nn.MaxPool2d(2, 2),
            ConvBlock(64, 128)
        )

        self.mask_branch = torch.nn.Sequential(
            ConvBlock(1, 8),
            torch.nn.MaxPool2d(2, 2),
            ConvBlock(8, 16),
            torch.nn.MaxPool2d(2, 2),
            ConvBlock(16, 32),
            torch.nn.MaxPool2d(2, 2),
            ConvBlock(32, 64)
        )

        self.image_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.mask_pool = torch.nn.AdaptiveAvgPool2d(1)

        self.fusion = torch.nn.Sequential(
            torch.nn.Linear(128 + 64, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.3)
        )

        self.classifier = torch.nn.Linear(128, num_classes)

    def forward(self, image, mask):
        image_features = self.image_branch(image)
        image_features = self.image_pool(image_features)
        image_features = torch.flatten(image_features, 1)

        mask_features = self.mask_branch(mask)
        mask_features = self.mask_pool(mask_features)
        mask_features = torch.flatten(mask_features, 1)

        fused_features = torch.cat([image_features, mask_features], dim=1)
        fused_features = self.fusion(fused_features)
        logits = self.classifier(fused_features)

        return logits



def load_classification_model():
    global CLASSIFICATION_MODEL
    if CLASSIFICATION_MODEL is not None:
        return CLASSIFICATION_MODEL

    model = SkinLesionFeatureFusionClassifier(num_classes=7)
    checkpoint = torch.load(CLASSIFICATION_CHECKPOINT, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(DEVICE)
    model.eval()
    CLASSIFICATION_MODEL = model

    print("Feature Fusion loaded successfully")
    print("Feature Fusion checkpoint epoch:", checkpoint["epoch"])
    print("Validation Macro F1:", checkpoint["val_macro_f1"])

    return model


COARSE_MODEL = None
CLASSIFICATION_MODEL = None

CLASS_NAMES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]



def prepare_image(image_path):
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise ValueError(f"无法读取图片，请检查文件格式或文件是否损坏: {image_path}") from e

    image_256 = image.resize(
        IMAGE_SIZE_SEG,
        Image.Resampling.BILINEAR
    )

    image_128 = image.resize(
        IMAGE_SIZE_CLS,
        Image.Resampling.BILINEAR
    )

    image_256 = np.array(image_256, dtype=np.float32) / 255.0
    image_128 = np.array(image_128, dtype=np.float32) / 255.0

    image_256 = torch.from_numpy(image_256).permute(2, 0, 1).unsqueeze(0)
    image_128 = torch.from_numpy(image_128).permute(2, 0, 1).unsqueeze(0)

    return image_256.to(DEVICE), image_128.to(DEVICE)



def predict_coarse_mask(model, image_256):
    with torch.no_grad():
        logits = model(image_256)
        mask = torch.argmax(logits, dim=1, keepdim=True).float()

    return mask



def resize_mask_for_classification(mask):
    mask = torch.nn.functional.interpolate(
        mask,
        size=IMAGE_SIZE_CLS,
        mode="nearest"
    )
    return mask



def predict_classification(model, image_128, mask_128):
    with torch.no_grad():
        logits = model(image_128, mask_128)
        probabilities = torch.softmax(logits, dim=1)
        predicted_index = torch.argmax(probabilities, dim=1).item()

    return predicted_index, probabilities[0]



def predict(image_path):
    coarse_model = load_coarse_model()
    classification_model = load_classification_model()

    image_256, image_128 = prepare_image(image_path)

    coarse_mask_256 = predict_coarse_mask(
        coarse_model,
        image_256
    )

    coarse_mask_128 = resize_mask_for_classification(
        coarse_mask_256
    )

    predicted_index, probabilities = predict_classification(
        classification_model,
        image_128,
        coarse_mask_128
    )

    predicted_class = CLASS_NAMES[predicted_index]

    return {
        "image_path": str(image_path),
        "predicted_class": predicted_class,
        "predicted_index": predicted_index,
        "probabilities": probabilities.cpu().numpy(),
        "coarse_mask_256": coarse_mask_256.cpu(),
        "coarse_mask_128": coarse_mask_128.cpu()
    }



def save_mask(mask, output_path):
    mask_image = (mask.squeeze().numpy() * 255).astype(np.uint8)
    Image.fromarray(mask_image).save(output_path)
    print("Mask saved to:", output_path)
