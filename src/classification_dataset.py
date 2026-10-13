import os
import pandas as pd
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class SkinLesionClassificationDataset(Dataset):
    """RGB-only ISIC 2018 Task 3 dataset; no segmentation masks are required."""

    def __init__(self, image_dir, label_csv, split_file, image_size=(128, 128)):
        self.image_dir = image_dir
        self.image_size = image_size
        labels_df = pd.read_csv(label_csv)
        self.labels = dict(zip(labels_df["image"], labels_df["label"]))

        with open(split_file, "r", encoding="utf-8") as f:
            self.image_names = [line.strip() for line in f if line.strip()]

        for name in self.image_names:
            image_path = os.path.join(self.image_dir, name + ".jpg")
            if not os.path.isfile(image_path):
                raise FileNotFoundError(f"找不到分类图片：{image_path}")
            if name not in self.labels:
                raise ValueError(f"标签 CSV 中找不到图片标签：{name}")

        print("分类数据集初始化完成")
        print("图片数量：", len(self.image_names))

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        name = self.image_names[index]
        image_path = os.path.join(self.image_dir, name + ".jpg")
        image = Image.open(image_path).convert("RGB")
        image = image.resize(self.image_size, Image.Resampling.BILINEAR)
        image = np.asarray(image, dtype=np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1).contiguous()
        label = torch.tensor(int(self.labels[name]), dtype=torch.long)
        return {"image": image, "label": label, "name": name}
