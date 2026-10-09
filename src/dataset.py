from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np


class ISICDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=(256, 256)):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_size = image_size

        self.images = sorted(self.image_dir.glob("*.jpg"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]

        mask_path = self.mask_dir / f"{image_path.stem}_segmentation.png"

        # 读取原始图片和 Mask
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Image 使用双线性插值
        image = image.resize(
            self.image_size,
            Image.Resampling.BILINEAR
        )

        # Mask 使用最近邻插值
        # 防止分割标签被插值成错误的灰度值
        mask = mask.resize(
            self.image_size,
            Image.Resampling.NEAREST
        )

        # Image 转换为 Tensor
        image = torch.from_numpy(
            np.array(image)
        ).permute(2, 0, 1).float() / 255.0

        # Mask 转换为 Tensor
        mask = torch.from_numpy(
            np.array(mask)
        ).unsqueeze(0).float() / 255.0

        return image, mask


if __name__ == "__main__":

    image_dir = (
        r"D:\GraduationProject\Skin-Lesion-Detection"
        r"\datasets\ISIC2018\segmentation"
        r"\ISIC2018_Task1-2_Training_Input"
    )

    mask_dir = (
        r"D:\GraduationProject\Skin-Lesion-Detection"
        r"\datasets\ISIC2018\segmentation"
        r"\ISIC2018_Task1_Training_GroundTruth"
    )

    dataset = ISICDataset(
        image_dir,
        mask_dir,
        image_size=(256, 256)
    )

    print("Dataset 总数量：", len(dataset))

    image, mask = dataset[0]

    print("Image Tensor shape：", image.shape)
    print("Mask Tensor shape：", mask.shape)

    print("Image 数据类型：", image.dtype)
    print("Mask 数据类型：", mask.dtype)

    print("Image 最小值：", image.min().item())
    print("Image 最大值：", image.max().item())

    print("Mask 最小值：", mask.min().item())
    print("Mask 最大值：", mask.max().item())