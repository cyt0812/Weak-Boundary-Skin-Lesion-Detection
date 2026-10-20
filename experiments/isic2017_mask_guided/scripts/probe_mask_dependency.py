"""Check whether the pilot Mask classifier actually uses its mask input.

Evaluates the same fixed checkpoint twice on ISIC2017 validation: once with
MALUNet-predicted masks and once with all-zero fourth channels. No retraining
or test-set access.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from pilot_mask_finetune import ControlledDataset
from train_classifier_ablation import Xception_dilation, evaluate


ROOT = Path(__file__).resolve().parent / "pilot_mask_optimization"
CHECKPOINT = ROOT / "mask" / "best.pth"


def main():
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model = Xception_dilation(input_channel=4, num_classes=3).cuda()
    model.load_state_dict(checkpoint["model"])
    results = {}
    predictions = {}
    for mode in ("mask", "zero"):
        dataset = ControlledDataset("val", mode)
        results[mode], rows = evaluate(
            model, DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0), with_predictions=True
        )
        predictions[mode] = {row[0]: row for row in rows}
    if predictions["mask"].keys() != predictions["zero"].keys():
        raise ValueError("Validation IDs differ between paired inputs")
    ids = list(predictions["mask"])
    changed = [image_id for image_id in ids if predictions["mask"][image_id][2] != predictions["zero"][image_id][2]]
    probability_differences = np.asarray([
        np.abs(np.asarray(predictions["mask"][image_id][3:]) - np.asarray(predictions["zero"][image_id][3:]))
        for image_id in ids
    ])
    kernel = checkpoint["model"]["conv1.weight"]
    summary = {
        "checkpoint_step": checkpoint["step"],
        "images": len(ids),
        "changed_prediction_count": len(changed),
        "changed_image_ids": changed,
        "mean_absolute_probability_difference": float(probability_differences.mean()),
        "max_absolute_probability_difference": float(probability_differences.max()),
        "mask_channel_kernel_norm": float(kernel[:, 3:].norm()),
        "rgb_kernel_norm": float(kernel[:, :3].norm()),
        "with_mask": {
            "accuracy": results["mask"]["accuracy"],
            "macro_f1": results["mask"]["macro_f1"],
            "melanoma_recall": results["mask"]["classification_report"]["melanoma"]["recall"],
        },
        "with_zero_channel": {
            "accuracy": results["zero"]["accuracy"],
            "macro_f1": results["zero"]["macro_f1"],
            "melanoma_recall": results["zero"]["classification_report"]["melanoma"]["recall"],
        },
    }
    path = ROOT / "mask_dependency_validation.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

