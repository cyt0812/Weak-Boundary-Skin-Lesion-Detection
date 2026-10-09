
from pathlib import Path
import sys
import torch
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

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "coarse_sn"

IMAGE_SIZE = (256, 256)
BATCH_SIZE = 2
NUM_WORKERS = 0
LEARNING_RATE = 1e-4
EPOCHS = 5
RESUME_CHECKPOINT = OUTPUT_DIR / "coarse_sn_epoch_4.pth"
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
    return ce + dice


def dice_score(logits, target):
    prediction = torch.argmax(logits, dim=1)
    prediction = prediction.float()
    target = target.float()

    smooth = 1e-6
    intersection = (prediction * target).sum(dim=(1, 2))
    denominator = prediction.sum(dim=(1, 2)) + target.sum(dim=(1, 2))

    dice = (2.0 * intersection + smooth) / (denominator + smooth)

    return dice.mean().item()


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

    print("========== CoarseSN 1 Epoch CPU 训练 ==========")
    print("设备：", DEVICE)
    print("输入尺寸：", IMAGE_SIZE)
    print("Batch Size：", BATCH_SIZE)
    print("Epochs：", EPOCHS)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
        num_workers=NUM_WORKERS,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        drop_last=True
    )

    model = load_pretrained_model()
    model = model.to(DEVICE)

    resume_checkpoint = torch.load(
        RESUME_CHECKPOINT,
        map_location="cpu"
    )

    model.load_state_dict(resume_checkpoint["model_state_dict"])

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])

    start_epoch = resume_checkpoint["epoch"]

    for epoch in range(start_epoch, EPOCHS):

        model.train()

        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.eval()

        train_loss = 0.0

        for batch_index, (images, masks, names) in enumerate(train_loader):

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad()

            logits = model(images)

            loss = combined_loss(logits, masks)

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

            if (batch_index + 1) % 100 == 0:
                print(
                    "Epoch",
                    epoch + 1,
                    "Batch",
                    batch_index + 1,
                    "/",
                    len(train_loader),
                    "Loss:",
                    loss.item()
                )

        train_loss = train_loss / len(train_loader)

        model.eval()

        val_loss = 0.0
        val_dice = 0.0

        with torch.no_grad():

            for images, masks, names in val_loader:

                images = images.to(DEVICE)
                masks = masks.to(DEVICE)

                logits = model(images)

                loss = combined_loss(logits, masks)

                val_loss += loss.item()

                val_dice += dice_score(logits, masks)

        val_loss = val_loss / len(val_loader)
        val_dice = val_dice / len(val_loader)

        print()
        print("========== Epoch", epoch + 1, "完成 ==========")
        print("Train Loss：", train_loss)
        print("Val Loss：", val_loss)
        print("Val Dice：", val_dice)

        save_path = OUTPUT_DIR / f"coarse_sn_epoch_{epoch + 1}.pth"

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_dice": val_dice
            },
            save_path
        )

        print("模型已保存：", save_path)

    print()
    print("========== 1 Epoch 测试完成 ==========")


if __name__ == "__main__":
    main()
