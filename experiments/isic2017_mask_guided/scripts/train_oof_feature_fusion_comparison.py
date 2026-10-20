"""Paired classifier comparison after the ISIC2017 OOF masks are complete.

Default is a read-only dry run. Each arm writes to a new, separate output path.
Official 600-image test data are not read here.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch

import train_feature_fusion_pilot as pilot
from prepare_oof_mask_folds import OUTPUT as PLAN_ROOT
from train_classifier_ablation import CROPS, MASKS, ClassDataset


ROOT = Path(__file__).resolve().parent
OOF_MASKS = Path(r"D:\Desktop\MALUNet-MB-DCNN-ISIC2017\predicted_masks_oof5_train")
OUTPUT = ROOT / "feature_fusion_oof_comparison"
TRAIN_MANIFEST = CROPS / "train" / "manifest.csv"


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SourceDataset(ClassDataset):
    train_mask_dir = MASKS / "train"

    def __init__(self, split, use_mask):
        super().__init__(split, use_mask)
        if split == "train" and use_mask:
            self.mask_dir = self.train_mask_dir
            missing = [row["file"] for row in self.rows if not (self.mask_dir / row["file"]).is_file()]
            if missing:
                raise FileNotFoundError(f"Missing {len(missing)} training masks in {self.mask_dir}")


def validate_oof_source():
    metadata_path = OOF_MASKS / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"OOF-mask generation is not complete: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("fold_manifest_sha256") != file_hash(PLAN_ROOT / "fold_assignments.csv"):
        raise ValueError("OOF-mask metadata does not match the current fold manifest")
    if metadata.get("crop_manifest_sha256") != file_hash(TRAIN_MANIFEST):
        raise ValueError("OOF-mask metadata does not match the current crop manifest")
    with TRAIN_MANIFEST.open(newline="", encoding="utf-8-sig") as file:
        names = [row["file"] for row in csv.DictReader(file)]
    if len(names) != 21768 or any(not (OOF_MASKS / name).is_file() for name in names):
        raise FileNotFoundError("OOF-mask directory is incomplete")
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("control", "in_sample_mask", "oof_mask"), required=True)
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Run one forward/backward batch without writing results")
    args = parser.parse_args()
    mode = "control" if args.variant == "control" else "mask"
    train_mask_dir = OOF_MASKS if args.variant == "oof_mask" else MASKS / "train"
    arm_root = OUTPUT / f"seed{args.seed}" / args.variant
    destination = arm_root / mode
    print(f"variant={args.variant}, seed={args.seed}, train_masks={train_mask_dir}")
    print(f"output={destination}; validation masks={MASKS / 'val'}; test set unused")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite a comparison arm: {destination}")
    if not args.execute:
        print("Dry run only. No training or output directory creation.")
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable")
    oof_metadata = validate_oof_source() if args.variant == "oof_mask" else None
    SourceDataset.train_mask_dir = train_mask_dir
    pilot.ClassDataset = SourceDataset
    pilot.OUTPUT = arm_root
    pilot.train_arm(
        mode=mode,
        steps=1000,
        eval_every=200,
        lr=0.0005,
        smoke=args.smoke,
        seed=args.seed,
        metadata_extra={
            "comparison_variant": args.variant,
            "training_mask_directory": str(train_mask_dir),
            "validation_mask_directory": str(MASKS / "val"),
            "oof_fold_manifest_sha256": oof_metadata["fold_manifest_sha256"] if oof_metadata else None,
            "official_test_used": False,
        },
    )


if __name__ == "__main__":
    main()

