"""Relate ISIC2017 validation Mask quality to classification effect.

Uses official validation segmentation ground truth and the best learned
feature-fusion Mask checkpoint. The same fixed classifier is evaluated with
predicted masks and with uniform pooling, so the difference isolates its
dependence on Mask input. Test data and checkpoints are untouched.
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

from train_classifier_ablation import ClassDataset
from train_feature_fusion_pilot import OUTPUT as MODEL_OUTPUT, FeatureFusionClassifier, evaluate


ROOT = Path(__file__).resolve().parent
QUALITY_CSV = ROOT / "diagnostics" / "val_crop_mask_quality.csv"
PAIRED_CSV = MODEL_OUTPUT / "paired_validation_predictions.csv"
OUTPUT = ROOT / "diagnostics"


def mask_quality():
    grouped = defaultdict(list)
    with QUALITY_CSV.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            grouped[row["image_id"]].append(row)
    if len(grouped) != 150 or any(len(rows) != 9 for rows in grouped.values()):
        raise ValueError("Expected nine Mask-quality crop rows for each of 150 validation images")
    result = {}
    for image_id, rows in grouped.items():
        original = next(row for row in rows if int(row["crop"]) == 0)
        result[image_id] = {
            "mean_crop_dice": float(np.mean([float(row["dice"]) for row in rows])),
            "full_image_dice": float(original["dice"]),
            "mean_predicted_coverage": float(np.mean([float(row["predicted_coverage"]) for row in rows])),
        }
    return result


def labels_and_control():
    with PAIRED_CSV.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != 150:
        raise ValueError(f"Expected 150 paired classification rows, got {len(rows)}")
    return {row["image_id"]: row for row in rows}


def fixed_model_predictions():
    saved = torch.load(MODEL_OUTPUT / "mask" / "best.pth", map_location="cpu", weights_only=True)
    model = FeatureFusionClassifier(use_mask=True).cuda()
    model.head.load_state_dict(saved["head"])
    loader = DataLoader(ClassDataset("val", use_mask=True), batch_size=16, shuffle=False, num_workers=0)
    with_mask, with_predictions = evaluate(model, loader, return_predictions=True)
    uniform, uniform_predictions = evaluate(model, loader, force_uniform=True, return_predictions=True)
    return with_mask, uniform, with_predictions, uniform_predictions


def group_summary(rows):
    if not rows:
        return {"images": 0, "mean_crop_dice": None, "mean_full_image_dice": None}
    return {
        "images": len(rows),
        "mean_crop_dice": float(np.mean([row["mean_crop_dice"] for row in rows])),
        "mean_full_image_dice": float(np.mean([row["full_image_dice"] for row in rows])),
        "image_ids": [row["image_id"] for row in rows],
    }


def correlation(rows):
    if len(rows) < 5:
        return None
    statistic = spearmanr(
        [row["mean_crop_dice"] for row in rows],
        [row["true_class_probability_change"] for row in rows],
    )
    return {"spearman_rho": float(statistic.statistic), "exploratory_p_value": float(statistic.pvalue)}


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable")
    quality = mask_quality()
    comparison = labels_and_control()
    with_metrics, uniform_metrics, with_predictions, uniform_predictions = fixed_model_predictions()
    if quality.keys() != comparison.keys() or quality.keys() != with_predictions.keys() or quality.keys() != uniform_predictions.keys():
        raise ValueError("Validation image IDs do not match")

    rows = []
    for image_id in comparison:
        paired = comparison[image_id]
        truth = int(paired["true_label"])
        mask_pred, mask_prob = with_predictions[image_id]
        uniform_pred, uniform_prob = uniform_predictions[image_id]
        if mask_pred != int(paired["mask_pred"]):
            raise ValueError(f"Mask prediction disagrees with previous paired CSV for {image_id}")
        control_pred = int(paired["control_pred"])
        row = {
            "image_id": image_id,
            "true_label": truth,
            "control_pred": control_pred,
            "mask_pred": mask_pred,
            "uniform_pred": uniform_pred,
            **quality[image_id],
            "true_class_probability_change": float(mask_prob[truth] - uniform_prob[truth]),
            "mask_correct": int(mask_pred == truth),
            "uniform_correct": int(uniform_pred == truth),
            "control_correct": int(control_pred == truth),
        }
        rows.append(row)

    groups = {
        "mask_helped_vs_uniform": [row for row in rows if row["mask_correct"] and not row["uniform_correct"]],
        "mask_hurt_vs_uniform": [row for row in rows if row["uniform_correct"] and not row["mask_correct"]],
        "both_correct_vs_uniform": [row for row in rows if row["mask_correct"] and row["uniform_correct"]],
        "both_wrong_vs_uniform": [row for row in rows if not row["mask_correct"] and not row["uniform_correct"]],
        "control_only_correct": [row for row in rows if row["control_correct"] and not row["mask_correct"]],
        "mask_only_correct": [row for row in rows if row["mask_correct"] and not row["control_correct"]],
    }
    result = {
        "images": len(rows),
        "fixed_mask_model_macro_f1_with_mask": with_metrics["macro_f1"],
        "fixed_mask_model_macro_f1_uniform": uniform_metrics["macro_f1"],
        "groups": {key: group_summary(value) for key, value in groups.items()},
        "correlation_mean_crop_dice_vs_true_class_probability_change": correlation(rows),
        "correlation_by_true_class": {
            str(label): correlation([row for row in rows if row["true_label"] == label])
            for label in range(3)
        },
        "limitation": "Exploratory validation analysis; correlations are not proof of causation and this split selected the pilot checkpoints.",
    }
    with (OUTPUT / "mask_quality_classification_effect.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT / "mask_quality_classification_effect.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

