"""Controlled ISIC2017 pilot: learned lesion-feature fusion vs no-mask control.

The same frozen RGB MB-DCNN/Xception backbone extracts spatial features from
the original image. Both arms train an identical small classification head on
global features plus a second pooled feature vector. The Mask arm pools the
second vector inside a MALUNet-predicted mask; the control pools it globally.
Only train/validation data are used. Original checkpoints are read-only.
"""

import argparse
import csv
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import DataLoader

from train_classifier_ablation import CROPS, OUTPUT as BASE_RESULTS, ClassDataset, Xception_dilation


BASE_CHECKPOINT = BASE_RESULTS / "rgb" / "best.pth"
OUTPUT = Path(__file__).resolve().parent / "feature_fusion_pilot"


class FeatureFusionClassifier(nn.Module):
    def __init__(self, use_mask):
        super().__init__()
        self.use_mask = use_mask
        self.backbone = Xception_dilation(input_channel=3, num_classes=3)
        saved = torch.load(BASE_CHECKPOINT, map_location="cpu", weights_only=True)
        self.backbone.load_state_dict(saved["model"])
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.backbone.eval()
        self.head = nn.Sequential(
            nn.LayerNorm(4096),
            nn.Dropout(0.30),
            nn.Linear(4096, 256),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(256, 3),
        )

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, inputs, force_uniform=False):
        rgb = inputs[:, :3]
        with torch.no_grad():
            self.backbone(rgb)
            spatial = self.backbone.layers[0]
            global_features = self.backbone.avgpool(spatial).flatten(1)
            if self.use_mask and not force_uniform:
                mask = F.interpolate(inputs[:, 3:4], size=spatial.shape[-2:], mode="nearest")
                lesion_features = (spatial * mask).sum(dim=(2, 3)) / mask.sum(dim=(2, 3)).clamp_min(1)
            else:
                lesion_features = global_features
            features = torch.cat([global_features, lesion_features], dim=1)
        return self.head(features)


def class_weights():
    with (CROPS / "train" / "manifest.csv").open(newline="", encoding="utf-8-sig") as file:
        counts = Counter(int(row["label"]) for row in csv.DictReader(file))
    if set(counts) != {0, 1, 2}:
        raise ValueError(f"Unexpected class counts: {counts}")
    weights = torch.tensor([counts[index] ** -0.5 for index in range(3)], dtype=torch.float32)
    return weights / weights.mean(), counts


def evaluate(model, loader, force_uniform=False, return_predictions=False):
    model.eval()
    scores = {}
    labels = {}
    with torch.inference_mode():
        for inputs, targets, image_ids in loader:
            with torch.autocast("cuda", dtype=torch.float16):
                output = model(inputs.cuda(non_blocking=True), force_uniform=force_uniform)
            probs = output.float().softmax(1).cpu().numpy()
            for image_id, target, probability in zip(image_ids, targets.tolist(), probs):
                scores.setdefault(image_id, []).append(probability)
                labels[image_id] = target
    ids = list(scores)
    if len(ids) != 150:
        raise ValueError(f"Expected 150 validation images, found {len(ids)}")
    truth = np.asarray([labels[image_id] for image_id in ids])
    probability = np.asarray([np.mean(scores[image_id], axis=0) for image_id in ids])
    predicted = probability.argmax(1)
    metrics = {
        "images": len(ids),
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, average="macro", labels=[0, 1, 2], zero_division=0)),
        "melanoma_recall": float(np.mean(predicted[truth == 1] == 1)),
        "confusion_matrix": confusion_matrix(truth, predicted, labels=[0, 1, 2]).tolist(),
        "auc_per_class": [float(roc_auc_score(truth == index, probability[:, index])) for index in range(3)],
    }
    if return_predictions:
        return metrics, {image_id: (int(pred), probability[number].tolist()) for number, (image_id, pred) in enumerate(zip(ids, predicted))}
    return metrics


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temp.replace(path)


def train_arm(mode, steps, eval_every, lr, smoke, seed=42, metadata_extra=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    use_mask = mode == "mask"
    model = FeatureFusionClassifier(use_mask).cuda()
    train_set = ClassDataset("train", use_mask=True)
    val_set = ClassDataset("val", use_mask=True)
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False, num_workers=0)
    weights, counts = class_weights()
    loss_fn = nn.CrossEntropyLoss(weight=weights.cuda(), label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=lr, weight_decay=1e-3)
    scaler = torch.amp.GradScaler("cuda")
    print(f"{mode}: crops train={len(train_set)} val={len(val_set)} counts={dict(counts)}", flush=True)
    iterator = iter(train_loader)
    if smoke:
        model.train()
        inputs, labels, _ = next(iterator)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = loss_fn(model(inputs.cuda()), labels.cuda())
        scaler.scale(loss).backward()
        print(f"{mode}: smoke loss={float(loss):.4f}", flush=True)
        return

    folder = OUTPUT / mode
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "status.json").exists() or (folder / "best.pth").exists():
        raise FileExistsError(f"Pilot output already exists: {folder}")
    metadata = {
        "mode": mode,
        "split": "official ISIC2017 train 2000 / val 150; test unused",
        "base_checkpoint": str(BASE_CHECKPOINT),
        "architecture": "frozen RGB Xception + learned head(global pooled, lesion pooled)",
        "second_pool": "MALUNet predicted mask" if use_mask else "uniform/global control",
        "steps": steps,
        "eval_every": eval_every,
        "initial_lr": lr,
        "head": "LayerNorm(4096), Dropout(.3), Linear(4096,256), ReLU, Dropout(.3), Linear(256,3)",
        "loss": "inverse-sqrt crop-count weighted cross entropy, label smoothing .05",
        "class_crop_counts": dict(counts),
        "class_weights": weights.tolist(),
        "seed": seed,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if metadata_extra:
        metadata.update(metadata_extra)
    write_json(folder / "metadata.json", metadata)
    best_score = (-1.0, -1.0)
    history = []
    for step in range(1, steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:
            group["lr"] = lr * (1 - (step - 1) / steps) ** 0.9
        loss_total = 0.0
        for _ in range(4):
            try:
                inputs, labels, _ = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                inputs, labels, _ = next(iterator)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = loss_fn(model(inputs.cuda(non_blocking=True)), labels.cuda(non_blocking=True))
            scaler.scale(loss / 4).backward()
            loss_total += float(loss) / 4
        scaler.step(optimizer)
        scaler.update()
        if step == 1 or step % 50 == 0:
            print(f"{mode}: step {step}/{steps} loss={loss_total:.4f}", flush=True)
        if step % eval_every == 0 or step == steps:
            metrics = evaluate(model, val_loader)
            score = (metrics["macro_f1"], metrics["melanoma_recall"])
            history.append({"step": step, "validation": metrics})
            if score > best_score:
                best_score = score
                temp = folder / "best.tmp"
                torch.save({"head": model.head.state_dict(), "step": step, "validation": metrics, "metadata": metadata}, temp)
                temp.replace(folder / "best.pth")
            write_json(folder / "history.json", history)
            write_json(folder / "status.json", {
                "stage": "complete" if step == steps else "training",
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "step": step,
                "total_steps": steps,
                "best_macro_f1": best_score[0],
                "best_melanoma_recall": best_score[1],
                "latest_validation": metrics,
            })
            print(f"{mode}: VAL step={step} macro_f1={score[0]:.4f} melanoma_recall={score[1]:.4f}", flush=True)

    if use_mask:
        best = torch.load(folder / "best.pth", map_location="cpu", weights_only=True)
        model.head.load_state_dict(best["head"])
        with_mask, mask_predictions = evaluate(model, val_loader, return_predictions=True)
        without_mask, zero_predictions = evaluate(model, val_loader, force_uniform=True, return_predictions=True)
        ids = list(mask_predictions)
        changes = [image_id for image_id in ids if mask_predictions[image_id][0] != zero_predictions[image_id][0]]
        mean_change = float(np.mean([
            np.abs(np.asarray(mask_predictions[image_id][1]) - np.asarray(zero_predictions[image_id][1])).mean()
            for image_id in ids
        ]))
        write_json(folder / "mask_dependency_validation.json", {
            "best_step": best["step"],
            "with_mask": with_mask,
            "mask_replaced_by_uniform": without_mask,
            "changed_predictions": len(changes),
            "changed_image_ids": changes,
            "mean_absolute_probability_change": mean_change,
        })
        print(f"mask dependency: {len(changes)} changed validation predictions; mean probability change={mean_change:.4f}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("control", "mask", "both"), default="both")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable")
    if args.steps <= 0 or args.eval_every <= 0 or args.lr <= 0:
        raise ValueError("Steps, eval interval, and learning rate must be positive")
    modes = ("control", "mask") if args.mode == "both" else (args.mode,)
    for mode in modes:
        train_arm(mode, args.steps, args.eval_every, args.lr, args.smoke)


if __name__ == "__main__":
    main()

