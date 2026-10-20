"""Validation-only probe of mild, explicit Mask-guided RGB input.

The existing RGB checkpoint is held fixed. The lesion pixels remain unchanged;
background pixels are multiplied by 1-alpha. This is an exploratory response
test, not a trained/final fusion result. The test set is never accessed.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from train_classifier_ablation import ClassDataset, OUTPUT as ORIGINAL_RESULTS, Xception_dilation, evaluate


OUTPUT = Path(__file__).resolve().parent / "diagnostics"
ALPHAS = (0.0, 0.25, 0.50)


class GuidedRGB(Dataset):
    def __init__(self, alpha):
        self.alpha = alpha
        self.base = ClassDataset("val", use_mask=True)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        four_channels, label, image_id = self.base[index]
        rgb, mask = four_channels[:3], four_channels[3:4]
        guided = rgb * (1 - self.alpha * (1 - mask))
        return guided, label, image_id


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable")
    checkpoint = torch.load(ORIGINAL_RESULTS / "rgb" / "best.pth", map_location="cpu", weights_only=True)
    model = Xception_dilation(input_channel=3, num_classes=3).cuda()
    model.load_state_dict(checkpoint["model"])
    result = {}
    for alpha in ALPHAS:
        metrics = evaluate(model, DataLoader(GuidedRGB(alpha), batch_size=8, shuffle=False, num_workers=0))
        result[f"{alpha:.2f}"] = {
            "images": metrics["images"],
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "melanoma_recall": metrics["classification_report"]["melanoma"]["recall"],
            "confusion_matrix": metrics["confusion_matrix"],
        }
        print(f"alpha={alpha:.2f} val Macro-F1={metrics['macro_f1']:.4f}", flush=True)
    if not np.isclose(result["0.00"]["macro_f1"], checkpoint["validation"]["macro_f1"]):
        raise ValueError("Alpha=0 baseline does not match saved checkpoint validation metric")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "validation_mask_gating_results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

