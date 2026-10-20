"""Validation-only analysis of crop policy with existing RGB/Mask checkpoints.

No training, no test-set inference, and no checkpoint modification. Each val
crop is inferred once; image-level predictions are averaged under fixed crop
policies to test whether tight crops are associated with the Mask drop.
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import DataLoader

from train_classifier_ablation import ClassDataset, OUTPUT as CHECKPOINTS, Xception_dilation


OUTPUT = Path(__file__).resolve().parent / "diagnostics"
POLICIES = {
    "all_9": set(range(9)),
    "without_tightest_4_8": set(range(9)) - {4, 8},
    "large_5": {0, 1, 2, 5, 6},
    "original_only": {0},
}


def crop_index(filename, image_id):
    if filename == f"{image_id}.png":
        return 0
    prefix = f"{image_id}_9_"
    stem = Path(filename).stem
    if not stem.startswith(prefix):
        raise ValueError(filename)
    return int(stem[len(prefix):])


def infer_mode(mode):
    use_mask = mode == "malunet_mask"
    dataset = ClassDataset("val", use_mask)
    model = Xception_dilation(input_channel=4 if use_mask else 3, num_classes=3).cuda()
    checkpoint = torch.load(CHECKPOINTS / mode / "best.pth", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    rows = []
    position = 0
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    with torch.inference_mode():
        for images, labels, ids in loader:
            probability = model(images.cuda(non_blocking=True)).softmax(1).cpu().numpy()
            for image_id, label, probs in zip(ids, labels.tolist(), probability):
                manifest = dataset.rows[position]
                if manifest["image_id"] != image_id or int(manifest["label"]) != label:
                    raise ValueError(f"Manifest mismatch at row {position}")
                rows.append({
                    "image_id": image_id,
                    "true_label": label,
                    "crop": crop_index(manifest["file"], image_id),
                    "prob_nevus": float(probs[0]),
                    "prob_melanoma": float(probs[1]),
                    "prob_seborrheic_keratosis": float(probs[2]),
                })
                position += 1
            if position % 300 < len(ids) or position == len(dataset):
                print(f"{mode}: {position}/{len(dataset)} crops", flush=True)
    del model
    torch.cuda.empty_cache()
    if len(rows) != 1350:
        raise ValueError(f"Expected 1350 crop predictions for {mode}, got {len(rows)}")
    return rows, checkpoint["validation"], checkpoint["step"]


def evaluate_policy(rows, allowed_crops):
    groups = defaultdict(list)
    labels = {}
    for row in rows:
        if row["crop"] in allowed_crops:
            image_id = row["image_id"]
            groups[image_id].append([
                row["prob_nevus"], row["prob_melanoma"], row["prob_seborrheic_keratosis"]
            ])
            labels[image_id] = row["true_label"]
    if len(groups) != 150 or any(len(items) != len(allowed_crops) for items in groups.values()):
        raise ValueError("Crop policy did not retain the same number of crops per image")
    ids = list(groups)
    y = np.asarray([labels[image_id] for image_id in ids])
    probabilities = np.asarray([np.mean(groups[image_id], axis=0) for image_id in ids])
    predicted = probabilities.argmax(1)
    return {
        "images": len(ids),
        "accuracy": float(accuracy_score(y, predicted)),
        "macro_f1": float(f1_score(y, predicted, labels=[0, 1, 2], average="macro", zero_division=0)),
        "melanoma_recall": float(np.mean(predicted[y == 1] == 1)),
        "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1, 2]).tolist(),
        "auc_per_class": [
            float(roc_auc_score(y == category, probabilities[:, category])) for category in range(3)
        ],
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable; this script intentionally avoids slow CPU inference")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for mode in ("rgb", "malunet_mask"):
        rows, checkpoint_validation, best_step = infer_mode(mode)
        with (OUTPUT / f"{mode}_val_crop_probabilities.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        metrics = {name: evaluate_policy(rows, crops) for name, crops in POLICIES.items()}
        all_9 = metrics["all_9"]
        if not np.isclose(all_9["accuracy"], checkpoint_validation["accuracy"]) or not np.isclose(
            all_9["macro_f1"], checkpoint_validation["macro_f1"]
        ):
            raise ValueError(f"Recomputed all-9 validation metrics differ from checkpoint for {mode}")
        summary[mode] = {"best_step": best_step, "policies": metrics}
        print(mode, json.dumps(metrics, indent=2), flush=True)
    (OUTPUT / "validation_crop_policy_results.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

