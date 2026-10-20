"""Run one MALUNet fold after explicit approval; default is a read-only dry run.

The held-out classification-train images are not used in segmentation
parameter training or checkpoint selection. Original results are protected by
using a separate, previously nonexistent result directory for each fold.
"""

import argparse
import csv
import ctypes
import os
from pathlib import Path

import torch
from torchvision import transforms

from prepare_oof_mask_folds import OUTPUT as PLAN_ROOT, SEGMENTATION_ROOT
from run_malunet_isic2017 import PairedDataset, RESULTS_ROOT, malunet_train, setting_config
from run_malunet_isic2017 import (
    myNormalize,
    myRandomHorizontalFlip,
    myRandomRotation,
    myRandomVerticalFlip,
    myResize,
    myToTensor,
)


ASSIGNMENTS = PLAN_ROOT / "fold_assignments.csv"
TRAIN_IDS = set()
HELD_OUT_IDS = set()


class FoldDataset(PairedDataset):
    def __init__(self, data_path, config, train=True):
        super().__init__(data_path, config, train=train)
        if train:
            self.pairs = [(image, mask) for image, mask in self.pairs if image.stem in TRAIN_IDS]
            observed = {image.stem for image, _ in self.pairs}
            if observed != TRAIN_IDS or observed & HELD_OUT_IDS:
                raise ValueError("Fold training data mismatch or held-out leakage")
            print(f"fold training: {len(self.pairs)} paired images; held out {len(HELD_OUT_IDS)}", flush=True)


def load_fold(fold):
    with ASSIGNMENTS.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != 2000 or len({row["image_id"] for row in rows}) != 2000:
        raise ValueError("Fold manifest must contain 2000 unique image IDs")
    expected = set(range(5))
    if {int(row["held_out_fold"]) for row in rows} != expected:
        raise ValueError("Fold manifest must contain folds 0 through 4")
    train_ids = {row["image_id"] for row in rows if int(row["held_out_fold"]) != fold}
    held_out_ids = {row["image_id"] for row in rows if int(row["held_out_fold"]) == fold}
    if len(train_ids) != 1600 or len(held_out_ids) != 400 or train_ids & held_out_ids:
        raise ValueError("Expected 1600 train and 400 held-out images")
    return train_ids, held_out_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--epochs", type=int, choices=(105, 300), default=105)
    parser.add_argument("--execute", action="store_true", help="Actually start this long training run")
    parser.add_argument("--resume", action="store_true", help="Resume an interrupted fold with its existing latest.pth")
    args = parser.parse_args()
    global TRAIN_IDS, HELD_OUT_IDS
    TRAIN_IDS, HELD_OUT_IDS = load_fold(args.fold)
    available = {path.stem for path in (SEGMENTATION_ROOT / "train" / "images").glob("ISIC_*.png")}
    if TRAIN_IDS | HELD_OUT_IDS != available:
        raise ValueError("Fold manifest and segmentation image directory differ")
    output = RESULTS_ROOT / f"oof5_fold{args.fold}_epochs{args.epochs}"
    if output.exists():
        if not args.resume:
            raise FileExistsError(f"Refusing to use an existing output directory: {output}")
        checkpoints = output / "checkpoints"
        if list(checkpoints.glob("best-epoch*-loss*.pth")):
            raise FileExistsError(f"Fold is already complete: {output}")
        if not (checkpoints / "latest.pth").is_file():
            raise FileNotFoundError(f"Cannot resume without latest.pth: {output}")
    elif args.resume:
        raise FileNotFoundError(f"No interrupted fold to resume: {output}")
    print(f"fold={args.fold}, training images={len(TRAIN_IDS)}, held-out images={len(HELD_OUT_IDS)}")
    print(f"epochs={args.epochs}, output={output}")
    print("Segmentation validation remains the official 150-image validation split; held-out fold is never used for checkpoint selection.")
    if not args.execute:
        print("Dry run only. No model training or output directory creation.")
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")

    setting_config.datasets = "isic17"
    setting_config.data_path = str(SEGMENTATION_ROOT) + os.sep
    setting_config.work_dir = str(output) + os.sep
    setting_config.epochs = args.epochs
    setting_config.val_interval = 30
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
    malunet_train.NPY_datasets = FoldDataset
    if not args.resume:
        output.mkdir(parents=True)
    es_continuous = 0x80000000
    es_system_required = 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | es_system_required)
    try:
        malunet_train.main(setting_config)
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(es_continuous)


if __name__ == "__main__":
    main()

