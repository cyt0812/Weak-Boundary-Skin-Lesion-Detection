import os
import pandas as pd
import torch
import numpy as np

from PIL import Image
from torch.utils.data import Dataset


class SkinLesionClassificationDataset(Dataset):

    def __init__(
        self,
        image_dir,
        mask_dir,
        label_csv,
        split_file,
        image_size=(128, 128)
    ):

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size

        # =========================
        # 读取标签
        # =========================

        labels_df = pd.read_csv(label_csv)

        # image -> label
        self.labels = dict(
            zip(
                labels_df["image"],
                labels_df["label"]
            )
        )

        # =========================
        # 读取数据划分
        # =========================

        with open(split_file, "r", encoding="utf-8") as f:

            self.image_names = [
                line.strip()
                for line in f
                if line.strip()
            ]

        # =========================
        # 检查文件
        # =========================

        valid_names = []

        for name in self.image_names:

            image_path = os.path.join(
                self.image_dir,
                name + ".jpg"
            )

            mask_path = os.path.join(
                self.mask_dir,
                name + "_coarse_mask.png"
            )

            if not os.path.exists(image_path):
                raise FileNotFoundError(
                    f"找不到图片:\n{image_path}"
                )

            if not os.path.exists(mask_path):
                raise FileNotFoundError(
                    f"找不到Coarse Mask:\n{mask_path}"
                )

            if name not in self.labels:
                raise ValueError(
                    f"找不到标签: {name}"
                )

            valid_names.append(name)

        self.image_names = valid_names

        print("Dataset 初始化完成")
        print("图片数量:", len(self.image_names))

    def __len__(self):

        return len(self.image_names)

    def __getitem__(self, index):

        name = self.image_names[index]

        # =========================
        # 图片路径
        # =========================

        image_path = os.path.join(
            self.image_dir,
            name + ".jpg"
        )

        mask_path = os.path.join(
            self.mask_dir,
            name + "_coarse_mask.png"
        )

        # =========================
        # 读取 RGB 图片
        # =========================

        image = Image.open(
            image_path
        ).convert("RGB")

        image = image.resize(
            self.image_size,
            Image.Resampling.BILINEAR
        )

        image = np.array(
            image
        ).astype(np.float32) / 255.0

        image = torch.from_numpy(
            image
        ).permute(2, 0, 1)

        # =========================
        # 读取 Coarse Mask
        # =========================

        mask = Image.open(
            mask_path
        ).convert("L")

        mask = mask.resize(
            self.image_size,
            Image.Resampling.NEAREST
        )

        mask = np.array(
            mask
        ).astype(np.float32) / 255.0

        mask = torch.from_numpy(
            mask
        ).unsqueeze(0)

        # =========================
        # RGB + Mask
        # =========================

        fusion_input = torch.cat(
            [image, mask],
            dim=0
        )

        # =========================
        # Label
        # =========================

        label = torch.tensor(
            self.labels[name],
            dtype=torch.long
        )

        return {
            "image": image,
            "mask": mask,
            "input": fusion_input,
            "label": label,
            "name": name
        }


# ============================================================
# 测试 Dataset
# ============================================================

if __name__ == "__main__":

    project_root = r"D:\GraduationProject\Skin-Lesion-Detection"

    image_dir = os.path.join(
        project_root,
        "datasets",
        "ISIC2018",
        "classification",
        "images",
        "ISIC2018_Task3_Training_Input"
    )

    mask_dir = os.path.join(
        project_root,
        "outputs",
        "classification_coarse_masks"
    )

    label_csv = os.path.join(
        project_root,
        "datasets",
        "ISIC2018",
        "classification",
        "splits",
        "classification_labels.csv"
    )

    split_file = os.path.join(
        project_root,
        "datasets",
        "ISIC2018",
        "classification",
        "splits",
        "train_cls.txt"
    )

    print("========== Classification Dataset 测试 ==========")

    dataset = SkinLesionClassificationDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        label_csv=label_csv,
        split_file=split_file
    )

    sample = dataset[0]

    print()
    print("第一张图片:")
    print(sample["name"])

    print()
    print("Image shape:")
    print(sample["image"].shape)

    print()
    print("Mask shape:")
    print(sample["mask"].shape)

    print()
    print("Fusion input shape:")
    print(sample["input"].shape)

    print()
    print("Label:")
    print(sample["label"])

    print()
    print("Input dtype:")
    print(sample["input"].dtype)

    print()
    print("Mask unique values:")
    print(torch.unique(sample["mask"]))

    print()
    print("==============================================")