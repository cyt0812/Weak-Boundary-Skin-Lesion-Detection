"""Run MALUNet on the official ISIC2017 train/validation segmentation split.

Leaves the original MALUNet code, ISIC2018 data, and checkpoints untouched.
"""

import argparse
import ctypes
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


MALUNET_PROJECT = Path(r"D:\Desktop\MALUNet-main\MALUNet-main")
ISIC2017_DATA = Path(r"D:\Desktop\MALUNet-ISIC2017\data\isic2017")
RESULTS_ROOT = Path(r"D:\Desktop\MALUNet-ISIC2017\results")
sys.path.insert(0, str(MALUNET_PROJECT))
os.chdir(MALUNET_PROJECT)

import train as malunet_train  # noqa: E402
from configs.config_setting import setting_config  # noqa: E402
from utils import (  # noqa: E402
    myNormalize,
    myRandomHorizontalFlip,
    myRandomRotation,
    myRandomVerticalFlip,
    myResize,
    myToTensor,
)


class PairedDataset(Dataset):
    """Pair images and masks by ISIC ID, not independent directory order."""

    def __init__(self, data_path, config, train=True):
        split = "train" if train else "val"
        root = Path(data_path) / split
        images = {p.stem: p for p in (root / "images").glob("ISIC_*.png")}
        masks = {p.stem: p for p in (root / "masks").glob("ISIC_*.png")}
        if not images or images.keys() != masks.keys():
            raise ValueError(f"{split}: image/mask IDs are missing or unmatched")
        self.pairs = [(images[isic_id], masks[isic_id]) for isic_id in sorted(images)]
        self.transformer = config.train_transformer if train else config.test_transformer
        print(f"{split}: {len(self.pairs)} verified pairs", flush=True)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, mask_path = self.pairs[index]
        with Image.open(image_path) as image, Image.open(mask_path) as mask:
            image_array = np.asarray(image.convert("RGB"))
            mask_array = np.asarray(mask.convert("L"), dtype=np.float32)[..., None] / 255.0
        return self.transformer((image_array, mask_array))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, choices=(1, 300), required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required by the original MALUNet train.py")

    output_name = "smoke_one_epoch" if args.epochs == 1 else "malunet_isic17_300_epochs"
    output = RESULTS_ROOT / output_name
    setting_config.datasets = "isic17"
    setting_config.data_path = str(ISIC2017_DATA) + os.sep
    setting_config.work_dir = str(output) + os.sep
    setting_config.epochs = args.epochs
    setting_config.val_interval = 1 if args.epochs == 1 else 30
    setting_config.train_transformer = transforms.Compose([
        myNormalize("isic17", train=True),
        myToTensor(),
        myRandomHorizontalFlip(p=0.5),
        myRandomVerticalFlip(p=0.5),
        myRandomRotation(p=0.5, degree=[0, 360]),
        myResize(256, 256),
    ])
    setting_config.test_transformer = transforms.Compose([
        myNormalize("isic17", train=False),
        myToTensor(),
        myResize(256, 256),
    ])
    malunet_train.NPY_datasets = PairedDataset
    output.mkdir(parents=True, exist_ok=True)

    # Allow the display to sleep, but keep the computer awake while training.
    es_continuous = 0x80000000
    es_system_required = 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | es_system_required)
    try:
        print(f"Starting MALUNet ISIC2017: epochs={args.epochs}, output={output}", flush=True)
        malunet_train.main(setting_config)
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(es_continuous)


if __name__ == "__main__":
    main()

