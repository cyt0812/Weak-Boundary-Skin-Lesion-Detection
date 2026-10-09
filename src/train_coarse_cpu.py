
from pathlib import Path
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

PROJECT_ROOT = Path(r"D:\GraduationProject\Skin-Lesion-Detection")
MB_DCNN_ROOT = Path(r"D:\GraduationProject\MB-DCNN")

IMAGE_DIR = PROJECT_ROOT / "datasets" / "ISIC2018" / "segmentation" / "ISIC2018_Task1-2_Training_Input"
MASK_DIR = PROJECT_ROOT / "datasets" / "ISIC2018" / "segmentation" / "ISIC2018_Task1_Training_GroundTruth"
SPLIT_DIR = PROJECT_ROOT / "datasets" / "ISIC2018" / "splits"
PRETRAINED = MB_DCNN_ROOT / "models" / "deeplabv3plus_xception_VOC2012_epoch46_all.pth"

IMAGE_SIZE = (256, 256)
BATCH_SIZE = 2
NUM_WORKERS = 0
LEARNING_RATE = 1e-4
DEVICE = torch.device("cpu")


class ISICCoarseDataset(Dataset):
    def __init__(self, image_dir, mask_dir, split_file, image_size=(256, 256)):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_size = image_size

        stems = Path(split_file).read_text(encoding="utf-8").splitlines()
        self.stems = [s.strip() for s in stems if s.strip()]

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, index):
        stem = self.stems[index]

        image_path = self.image_dir / f"{stem}.jpg"
        mask_path = self.mask_dir / f"{stem}_segmentation.png"

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = image.resize(self.image_size, Image.Resampling.BILINEAR)
        mask = mask.resize(self.image_size, Image.Resampling.NEAREST)

        image = np.array(image, dtype=np.float32) / 255.0
        mask = np.array(mask, dtype=np.uint8)

        mask = (mask > 127).astype(np.int64)

        image = torch.from_numpy(image).permute(2, 0, 1)
        mask = torch.from_numpy(mask)

        return image, mask, stem


def dice_loss(logits, target):
    probability = torch.softmax(logits, dim=1)[:, 1]
    target = target.float()

    smooth = 1e-6
    intersection = (probability * target).sum(dim=(1, 2))
    denominator = probability.sum(dim=(1, 2)) + target.sum(dim=(1, 2))

    dice = (2.0 * intersection + smooth) / (denominator + smooth)

    return 1.0 - dice.mean()


def combined_loss(logits, target):
    ce = F.cross_entropy(logits, target)
    dice = dice_loss(logits, target)
    return ce + dice, ce.detach(), dice.detach()


def load_pretrained_model():
    sys.path.insert(0, str(MB_DCNN_ROOT))

    from net.models import deeplabv3plus

    model = deeplabv3plus(num_classes=2)

    checkpoint = torch.load(
        PRETRAINED,
        map_location="cpu"
    )

    checkpoint = {
        key.replace("module.", "", 1): value
        for key, value in checkpoint.items()
        if key.replace("module.", "", 1)
        not in ["cls_conv.weight", "cls_conv.bias"]
    }

    result = model.load_state_dict(checkpoint, strict=False)

    print("模型创建成功")
    print("Missing keys：", result.missing_keys)
    print("Unexpected keys：", result.unexpected_keys)

    return model


def main():
    print("========== CoarseSN CPU 训练测试 ==========")
    print("设备：", DEVICE)
    print("输入尺寸：", IMAGE_SIZE)
    print("Batch Size：", BATCH_SIZE)

    train_dataset = ISICCoarseDataset(
        IMAGE_DIR,
        MASK_DIR,
        SPLIT_DIR / "train.txt",
        IMAGE_SIZE
    )

    val_dataset = ISICCoarseDataset(
        IMAGE_DIR,
        MASK_DIR,
        SPLIT_DIR / "val.txt",
        IMAGE_SIZE
    )

    print("训练集数量：", len(train_dataset))
    print("验证集数量：", len(val_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    model = load_pretrained_model()
    model = model.to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    model.train()

    images, masks, names = next(iter(train_loader))

    images = images.to(DEVICE)
    masks = masks.to(DEVICE)

    print("第一批图片 shape：", images.shape)
    print("第一批 Mask shape：", masks.shape)

    optimizer.zero_grad()

    logits = model(images)

    print("模型输出 shape：", logits.shape)

    loss, ce, dice = combined_loss(logits, masks)

    print("Cross Entropy Loss：", ce.item())
    print("Dice Loss：", dice.item())
    print("Total Loss：", loss.item())

    loss.backward()

    optimizer.step()

    print("反向传播成功")
    print("Optimizer 更新成功")
    print("========== 1 个 Batch 测试完成 ==========")


if __name__ == "__main__":
    main()
