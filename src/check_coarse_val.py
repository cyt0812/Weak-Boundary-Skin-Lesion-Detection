
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

CHECKPOINT = PROJECT_ROOT / "outputs" / "coarse_sn" / "coarse_sn_epoch_1.pth"

IMAGE_SIZE = (256, 256)
BATCH_SIZE = 2
NUM_WORKERS = 0
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


def dice_score(logits, target):
    prediction = torch.argmax(logits, dim=1)

    prediction = prediction.float()
    target = target.float()

    smooth = 1e-6

    intersection = (prediction * target).sum(dim=(1, 2))
    denominator = prediction.sum(dim=(1, 2)) + target.sum(dim=(1, 2))

    dice = (2.0 * intersection + smooth) / (denominator + smooth)

    return dice.mean().item()


def main():

    print("========== CoarseSN 验证集诊断 ==========")

    sys.path.insert(0, str(MB_DCNN_ROOT))

    from net.models import deeplabv3plus

    model = deeplabv3plus(num_classes=2)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu"
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(DEVICE)
    model.eval()

    print("模型加载成功")

    val_dataset = ISICCoarseDataset(
        IMAGE_DIR,
        MASK_DIR,
        SPLIT_DIR / "val.txt",
        IMAGE_SIZE
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        drop_last=False
    )

    print("验证集数量：", len(val_dataset))
    print("验证 Batch 数量：", len(val_loader))

    total_loss = 0.0
    total_ce = 0.0
    total_dice_loss = 0.0
    total_dice = 0.0

    total_pred_foreground = 0
    total_gt_foreground = 0
    total_pixels = 0

    min_logit = float("inf")
    max_logit = float("-inf")

    with torch.no_grad():

        for batch_index, (images, masks, names) in enumerate(val_loader):

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            logits = model(images)

            ce = F.cross_entropy(logits, masks)
            dl = dice_loss(logits, masks)
            loss = ce + dl

            ds = dice_score(logits, masks)

            prediction = torch.argmax(logits, dim=1)

            pred_foreground = (prediction == 1).sum().item()
            gt_foreground = (masks == 1).sum().item()
            pixels = masks.numel()

            total_loss += loss.item()
            total_ce += ce.item()
            total_dice_loss += dl.item()
            total_dice += ds

            total_pred_foreground += pred_foreground
            total_gt_foreground += gt_foreground
            total_pixels += pixels

            min_logit = min(min_logit, logits.min().item())
            max_logit = max(max_logit, logits.max().item())

            if batch_index < 5:

                print()
                print("Batch", batch_index + 1)
                print("图片：", names)
                print("CE Loss：", ce.item())
                print("Dice Loss：", dl.item())
                print("Total Loss：", loss.item())
                print("Dice：", ds)
                print("预测前景比例：", pred_foreground / pixels)
                print("GT 前景比例：", gt_foreground / pixels)
                print("Logit 最小值：", logits.min().item())
                print("Logit 最大值：", logits.max().item())

    n = len(val_loader)

    print()
    print("========== 验证集诊断结果 ==========")
    print("平均 CE Loss：", total_ce / n)
    print("平均 Dice Loss：", total_dice_loss / n)
    print("平均 Total Loss：", total_loss / n)
    print("平均 Dice：", total_dice / n)

    print()
    print("预测前景比例：", total_pred_foreground / total_pixels)
    print("GT 前景比例：", total_gt_foreground / total_pixels)

    print()
    print("Logit 最小值：", min_logit)
    print("Logit 最大值：", max_logit)

    print()
    print("========== 诊断完成 ==========")


if __name__ == "__main__":
    main()
