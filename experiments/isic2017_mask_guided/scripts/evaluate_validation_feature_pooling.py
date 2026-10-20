"""Validation-only feature-level MALUNet-mask guidance with fixed RGB weights.

The MB-DCNN Xception RGB backbone is unchanged. Predicted masks reweight its
last spatial feature map before the existing classifier head. Alpha=0 is
identical to the saved RGB model; positive alphas test a mid/late-fusion idea.
No retraining and no test-set access.
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_classifier_ablation import ClassDataset, OUTPUT as ORIGINAL_RESULTS, Xception_dilation, evaluate


OUTPUT = Path(__file__).resolve().parent / "diagnostics"
ALPHAS = (0.0, 0.10, 0.25, 0.50)


class MaskFeaturePool(Xception_dilation):
    def __init__(self):
        super().__init__(input_channel=3, num_classes=3)
        self.alpha = 0.0

    def forward(self, four_channels):
        rgb, mask = four_channels[:, :3], four_channels[:, 3:4]
        original_logits = super().forward(rgb)
        if self.alpha == 0:
            return original_logits
        features = self.layers[0]
        global_pool = self.avgpool(features).flatten(1)
        spatial_mask = F.interpolate(mask, size=features.shape[-2:], mode="nearest")
        lesion_pool = (features * spatial_mask).sum(dim=(2, 3)) / spatial_mask.sum(dim=(2, 3)).clamp_min(1)
        fused = (1 - self.alpha) * global_pool + self.alpha * lesion_pool
        return self.cls(fused)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable")
    checkpoint = torch.load(ORIGINAL_RESULTS / "rgb" / "best.pth", map_location="cpu", weights_only=True)
    model = MaskFeaturePool().cuda()
    model.load_state_dict(checkpoint["model"])
    loader = DataLoader(ClassDataset("val", use_mask=True), batch_size=8, shuffle=False, num_workers=0)
    result = {}
    for alpha in ALPHAS:
        model.alpha = alpha
        metrics = evaluate(model, loader)
        result[f"{alpha:.2f}"] = {
            "images": metrics["images"],
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "melanoma_recall": metrics["classification_report"]["melanoma"]["recall"],
            "confusion_matrix": metrics["confusion_matrix"],
        }
        print(f"alpha={alpha:.2f} val Macro-F1={metrics['macro_f1']:.4f}", flush=True)
    if not np.isclose(result["0.00"]["macro_f1"], checkpoint["validation"]["macro_f1"]):
        raise ValueError("Alpha=0 feature-pooling baseline differs from saved RGB validation")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "validation_feature_pooling_results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

