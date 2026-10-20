"""Paired validation comparison of frozen-backbone feature-fusion pilot arms.

Runs each already-selected best head on the same 150 ISIC2017 validation IDs,
then computes a class-stratified paired bootstrap of Mask-minus-control
Macro-F1. This is exploratory because both checkpoints were selected on the
same validation set; it is not an independent final confidence interval.
"""

import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from train_classifier_ablation import ClassDataset
from train_feature_fusion_pilot import OUTPUT, FeatureFusionClassifier, evaluate


def load_predictions(mode, loader):
    checkpoint = torch.load(OUTPUT / mode / "best.pth", map_location="cpu", weights_only=True)
    model = FeatureFusionClassifier(use_mask=(mode == "mask")).cuda()
    model.head.load_state_dict(checkpoint["head"])
    metrics, rows = evaluate(model, loader, return_predictions=True)
    if not np.isclose(metrics["macro_f1"], checkpoint["validation"]["macro_f1"]):
        raise ValueError(f"Saved validation score does not reproduce for {mode}")
    return checkpoint["step"], metrics, rows


def main():
    dataset = ClassDataset("val", use_mask=True)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    labels = {row["image_id"]: int(row["label"]) for row in dataset.rows}
    control_step, control_metrics, control = load_predictions("control", loader)
    mask_step, mask_metrics, mask = load_predictions("mask", loader)
    if control.keys() != mask.keys() or control.keys() != labels.keys():
        raise ValueError("Validation image IDs differ between arms")
    ids = list(control)
    y = np.asarray([labels[image_id] for image_id in ids])
    c = np.asarray([control[image_id][0] for image_id in ids])
    m = np.asarray([mask[image_id][0] for image_id in ids])
    rng = np.random.default_rng(42)
    indices_by_class = [np.flatnonzero(y == label) for label in range(3)]
    differences = []
    for _ in range(3000):
        sample = np.concatenate([rng.choice(indices, size=len(indices), replace=True) for indices in indices_by_class])
        differences.append(
            f1_score(y[sample], m[sample], labels=[0, 1, 2], average="macro", zero_division=0)
            - f1_score(y[sample], c[sample], labels=[0, 1, 2], average="macro", zero_division=0)
        )
    differences = np.asarray(differences)
    with (OUTPUT / "paired_validation_predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "image_id", "true_label", "control_pred", "mask_pred",
            "control_prob_nevus", "control_prob_melanoma", "control_prob_sk",
            "mask_prob_nevus", "mask_prob_melanoma", "mask_prob_sk",
        ])
        for image_id in ids:
            writer.writerow([image_id, labels[image_id], control[image_id][0], mask[image_id][0], *control[image_id][1], *mask[image_id][1]])
    summary = {
        "images": len(ids),
        "control_best_step": control_step,
        "mask_best_step": mask_step,
        "control_macro_f1": control_metrics["macro_f1"],
        "mask_macro_f1": mask_metrics["macro_f1"],
        "observed_mask_minus_control_macro_f1": mask_metrics["macro_f1"] - control_metrics["macro_f1"],
        "both_correct": int(np.sum((c == y) & (m == y))),
        "control_only_correct": int(np.sum((c == y) & (m != y))),
        "mask_only_correct": int(np.sum((c != y) & (m == y))),
        "both_wrong": int(np.sum((c != y) & (m != y))),
        "bootstrap": {
            "method": "3000 paired, class-stratified resamples of the same validation IDs",
            "difference_95_percentile_interval": [float(v) for v in np.quantile(differences, [0.025, 0.975])],
            "fraction_mask_greater_than_control": float(np.mean(differences > 0)),
            "warning": "Exploratory only: both best checkpoints were selected on this validation set.",
        },
    }
    (OUTPUT / "paired_validation_comparison.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

