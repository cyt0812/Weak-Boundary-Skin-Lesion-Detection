"""Verify OOF and original predicted masks are not accidentally identical."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from prepare_oof_mask_folds import OUTPUT as PLAN_ROOT
from train_classifier_ablation import MASKS


OOF_MASKS = Path(r"D:\Desktop\MALUNet-MB-DCNN-ISIC2017\predicted_masks_oof5_train")


def read_mask(path):
    with Image.open(path) as source:
        return np.asarray(source.convert("L")) > 127


def main():
    with (PLAN_ROOT / "fold_assignments.csv").open(newline="", encoding="utf-8") as file:
        assignments = list(csv.DictReader(file))
    if len(assignments) != 2000:
        raise ValueError("Expected 2000 assignments")
    by_class = defaultdict(list)
    rows = []
    for assignment in assignments:
        image_id = assignment["image_id"]
        old = read_mask(MASKS / "train" / f"{image_id}.png")
        new = read_mask(OOF_MASKS / f"{image_id}.png")
        if old.shape != new.shape or old.shape != (224, 224):
            raise ValueError(f"Wrong mask dimensions for {image_id}")
        row = {
            "image_id": image_id,
            "class_label": int(assignment["class_label"]),
            "pixel_disagreement_fraction": float(np.mean(old != new)),
        }
        rows.append(row)
        by_class[row["class_label"]].append(row)
    result = {
        "images": len(rows),
        "images_with_any_changed_mask_pixel": sum(row["pixel_disagreement_fraction"] > 0 for row in rows),
        "mean_pixel_disagreement_fraction": float(np.mean([row["pixel_disagreement_fraction"] for row in rows])),
        "median_pixel_disagreement_fraction": float(np.median([row["pixel_disagreement_fraction"] for row in rows])),
        "by_class_mean_pixel_disagreement_fraction": {
            str(label): float(np.mean([row["pixel_disagreement_fraction"] for row in group]))
            for label, group in sorted(by_class.items())
        },
    }
    (PLAN_ROOT / "oof_mask_difference_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

