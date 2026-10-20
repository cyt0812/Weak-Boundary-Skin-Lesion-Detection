"""Validation-only oracle-mask probe for a fixed learned fusion checkpoint.

Replace the MALUNet-predicted classification-crop masks with geometrically
matched official validation ground truth. This is a diagnostic, not a usable
classification pipeline or an independent test result. No training occurs.
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from analyze_validation_crop_masks import GROUND_TRUTH, ground_truth_crop
from train_classifier_ablation import CROPS, ClassDataset
from train_feature_fusion_pilot import OUTPUT as PILOT_OUTPUT, FeatureFusionClassifier, evaluate


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "diagnostics"


def crop_index(image_id, filename):
    if filename == f"{image_id}.png":
        return 0
    prefix = f"{image_id}_9_"
    stem = Path(filename).stem
    if not stem.startswith(prefix):
        raise ValueError(f"Unexpected crop filename: {filename}")
    index = int(stem[len(prefix):])
    if index not in range(1, 9):
        raise ValueError(f"Unexpected crop index: {filename}")
    return index


class OracleValidationDataset(Dataset):
    def __init__(self):
        with (CROPS / "val" / "manifest.csv").open(newline="", encoding="utf-8-sig") as file:
            self.rows = list(csv.DictReader(file))
        if len(self.rows) != 1350:
            raise ValueError(f"Expected 1350 validation crops, found {len(self.rows)}")
        by_image = defaultdict(list)
        for position, row in enumerate(self.rows):
            by_image[row["image_id"]].append((position, row))
        if len(by_image) != 150 or any(len(items) != 9 for items in by_image.values()):
            raise ValueError("Expected 150 validation images with nine crops each")
        self.masks = [None] * len(self.rows)
        for image_id, items in by_image.items():
            path = GROUND_TRUTH / f"{image_id}_segmentation.png"
            with Image.open(path) as source:
                truth = source.convert("L")
                for position, row in items:
                    index = crop_index(image_id, row["file"])
                    mask = ground_truth_crop(truth, index)
                    self.masks[position] = np.asarray(mask) > 127

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(CROPS / "val" / row["file"]) as source:
            rgb = np.asarray(source.convert("RGB"), dtype=np.float32).copy() / 255.0
        rgb = torch.from_numpy(rgb).permute(2, 0, 1)
        mask = torch.from_numpy(self.masks[index].astype(np.float32)).unsqueeze(0)
        return torch.cat([rgb, mask], dim=0), int(row["label"]), row["image_id"]


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable")
    saved = torch.load(PILOT_OUTPUT / "mask" / "best.pth", map_location="cpu", weights_only=True)
    model = FeatureFusionClassifier(use_mask=True).cuda()
    model.head.load_state_dict(saved["head"])
    predicted_loader = DataLoader(ClassDataset("val", use_mask=True), batch_size=16, shuffle=False, num_workers=0)
    oracle_loader = DataLoader(OracleValidationDataset(), batch_size=16, shuffle=False, num_workers=0)
    predicted_metrics, predicted = evaluate(model, predicted_loader, return_predictions=True)
    oracle_metrics, oracle = evaluate(model, oracle_loader, return_predictions=True)
    if predicted.keys() != oracle.keys():
        raise ValueError("Predicted-mask and oracle-mask image IDs differ")
    with (PILOT_OUTPUT / "paired_validation_predictions.csv").open(newline="", encoding="utf-8") as file:
        paired = {row["image_id"]: row for row in csv.DictReader(file)}
    if predicted.keys() != paired.keys():
        raise ValueError("Pilot comparison image IDs differ")

    rows = []
    for image_id, prior in paired.items():
        truth = int(prior["true_label"])
        predicted_label, predicted_probability = predicted[image_id]
        oracle_label, oracle_probability = oracle[image_id]
        if predicted_label != int(prior["mask_pred"]):
            raise ValueError(f"Predicted-mask label changed for {image_id}")
        rows.append({
            "image_id": image_id,
            "true_label": truth,
            "control_pred": int(prior["control_pred"]),
            "predicted_mask_pred": predicted_label,
            "oracle_mask_pred": oracle_label,
            "predicted_mask_true_probability": float(predicted_probability[truth]),
            "oracle_mask_true_probability": float(oracle_probability[truth]),
            "true_probability_change": float(oracle_probability[truth] - predicted_probability[truth]),
        })
    result = {
        "images": len(rows),
        "fixed_mask_checkpoint_step": int(saved["step"]),
        "predicted_mask": predicted_metrics,
        "oracle_ground_truth_mask": oracle_metrics,
        "prediction_changes": int(sum(row["predicted_mask_pred"] != row["oracle_mask_pred"] for row in rows)),
        "oracle_only_correct": [row["image_id"] for row in rows if row["oracle_mask_pred"] == row["true_label"] and row["predicted_mask_pred"] != row["true_label"]],
        "predicted_only_correct": [row["image_id"] for row in rows if row["predicted_mask_pred"] == row["true_label"] and row["oracle_mask_pred"] != row["true_label"]],
        "mean_true_probability_change": float(np.mean([row["true_probability_change"] for row in rows])),
        "limitation": "Oracle uses official validation segmentation labels at inference, is not deployable, and was explored on the checkpoint-selection validation split; it is not a strict upper bound or independent test.",
    }
    with (OUTPUT / "oracle_mask_probe.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT / "oracle_mask_probe.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

