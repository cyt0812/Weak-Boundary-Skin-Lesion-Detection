"""Exploratory paired validation comparison of old and OOF mask training.

All three heads use the same frozen RGB backbone and the same official
ISIC2017 validation images/predicted masks. The 600-image test set is unused.
"""

import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from train_classifier_ablation import ClassDataset
from train_feature_fusion_pilot import OUTPUT as ORIGINAL_OUTPUT, FeatureFusionClassifier, evaluate


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "feature_fusion_oof_comparison" / "seed42"
CHECKPOINTS = {
    "control": ORIGINAL_OUTPUT / "control" / "best.pth",
    "in_sample_mask": ORIGINAL_OUTPUT / "mask" / "best.pth",
    "oof_mask": OUTPUT / "oof_mask" / "mask" / "best.pth",
}


def evaluate_checkpoint(name, path, loader):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = FeatureFusionClassifier(use_mask=(name != "control")).cuda()
    model.head.load_state_dict(checkpoint["head"])
    metrics, predictions = evaluate(model, loader, return_predictions=True)
    if not np.isclose(metrics["macro_f1"], checkpoint["validation"]["macro_f1"]):
        raise ValueError(f"Stored validation score did not reproduce for {name}")
    return {"step": int(checkpoint["step"]), "metrics": metrics, "predictions": predictions}


def paired_summary(truth, reference, candidate, rng):
    ref = np.asarray(reference)
    new = np.asarray(candidate)
    y = np.asarray(truth)
    difference = f1_score(y, new, labels=[0, 1, 2], average="macro", zero_division=0) - f1_score(
        y, ref, labels=[0, 1, 2], average="macro", zero_division=0
    )
    indices_by_class = [np.flatnonzero(y == label) for label in range(3)]
    bootstrap = []
    for _ in range(3000):
        sample = np.concatenate([
            rng.choice(indices, size=len(indices), replace=True) for indices in indices_by_class
        ])
        bootstrap.append(
            f1_score(y[sample], new[sample], labels=[0, 1, 2], average="macro", zero_division=0)
            - f1_score(y[sample], ref[sample], labels=[0, 1, 2], average="macro", zero_division=0)
        )
    return {
        "macro_f1_difference": float(difference),
        "both_correct": int(np.sum((ref == y) & (new == y))),
        "reference_only_correct": int(np.sum((ref == y) & (new != y))),
        "candidate_only_correct": int(np.sum((ref != y) & (new == y))),
        "both_wrong": int(np.sum((ref != y) & (new != y))),
        "exploratory_paired_bootstrap_95_percentile_interval": [float(value) for value in np.quantile(bootstrap, [0.025, 0.975])],
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable")
    status = OUTPUT / "oof_mask" / "mask" / "status.json"
    if not status.is_file() or json.loads(status.read_text(encoding="utf-8")).get("stage") != "complete":
        raise RuntimeError("OOF-mask classifier training is not complete")
    dataset = ClassDataset("val", use_mask=True)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    labels = {row["image_id"]: int(row["label"]) for row in dataset.rows}
    arms = {name: evaluate_checkpoint(name, path, loader) for name, path in CHECKPOINTS.items()}
    ids = list(arms["control"]["predictions"])
    if len(ids) != 150 or set(ids) != set(labels) or any(set(arm["predictions"]) != set(ids) for arm in arms.values()):
        raise ValueError("Validation image IDs differ between arms")
    truth = np.asarray([labels[image_id] for image_id in ids])
    predicted = {
        name: np.asarray([arm["predictions"][image_id][0] for image_id in ids])
        for name, arm in arms.items()
    }
    rng = np.random.default_rng(42)
    summary = {
        "images": len(ids),
        "arms": {name: {"best_step": arm["step"], "validation_metrics": arm["metrics"]} for name, arm in arms.items()},
        "oof_minus_in_sample": paired_summary(truth, predicted["in_sample_mask"], predicted["oof_mask"], rng),
        "oof_minus_control": paired_summary(truth, predicted["control"], predicted["oof_mask"], rng),
        "warning": "Exploratory comparison on the same 150-image validation set used for checkpoint selection and earlier analyses; no official test inference. Original and OOF mask heads each trained once with seed 42 and the same settings, but only the training mask source differs.",
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "paired_validation_predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image_id", "true_label", "control_pred", "in_sample_mask_pred", "oof_mask_pred"])
        for image_id in ids:
            writer.writerow([
                image_id, labels[image_id],
                arms["control"]["predictions"][image_id][0],
                arms["in_sample_mask"]["predictions"][image_id][0],
                arms["oof_mask"]["predictions"][image_id][0],
            ])
    (OUTPUT / "paired_validation_comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

