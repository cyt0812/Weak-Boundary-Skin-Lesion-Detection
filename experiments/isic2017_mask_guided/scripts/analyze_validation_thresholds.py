"""Exploratory melanoma-threshold analysis on validation probabilities only.

Uses fixed candidate thresholds to diagnose decision bias. It does not touch
the test set or trained checkpoints. A threshold chosen here needs independent
confirmation before being reported as final performance.
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score


INPUT = Path(__file__).resolve().parent / "diagnostics"
THRESHOLDS = (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)


def image_probabilities(mode):
    with (INPUT / f"{mode}_val_crop_probabilities.csv").open(newline="", encoding="utf-8") as file:
        crops = list(csv.DictReader(file))
    grouped = defaultdict(list)
    labels = {}
    for crop in crops:
        image_id = crop["image_id"]
        grouped[image_id].append([
            float(crop["prob_nevus"]),
            float(crop["prob_melanoma"]),
            float(crop["prob_seborrheic_keratosis"]),
        ])
        labels[image_id] = int(crop["true_label"])
    if len(grouped) != 150 or any(len(value) != 9 for value in grouped.values()):
        raise ValueError(f"Expected 150 images with nine crops each for {mode}")
    ids = list(grouped)
    y = np.asarray([labels[image_id] for image_id in ids])
    probability = np.asarray([np.mean(grouped[image_id], axis=0) for image_id in ids])
    return y, probability


def score(y, predicted):
    return {
        "accuracy": float(accuracy_score(y, predicted)),
        "macro_f1": float(f1_score(y, predicted, average="macro", labels=[0, 1, 2])),
        "melanoma_recall": float(recall_score(y == 1, predicted == 1)),
        "predicted_melanomas": int(np.sum(predicted == 1)),
    }


def main():
    results = {}
    for mode in ("rgb", "malunet_mask"):
        y, probability = image_probabilities(mode)
        baseline = probability.argmax(1)
        result = {"argmax": score(y, baseline), "thresholds": {}}
        for threshold in THRESHOLDS:
            non_melanoma = np.where(probability[:, 0] >= probability[:, 2], 0, 2)
            predicted = np.where(probability[:, 1] >= threshold, 1, non_melanoma)
            result["thresholds"][f"{threshold:.2f}"] = score(y, predicted)
        results[mode] = result
    (INPUT / "validation_melanoma_thresholds.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

