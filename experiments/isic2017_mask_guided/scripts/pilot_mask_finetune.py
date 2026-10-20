"""Small, validation-only controlled pilot for MALUNet-mask guidance.

Both arms start from the exact same trained RGB classifier, expanded to four
channels with a zero-initialized fourth-channel kernel. The control always
receives a zero fourth channel; the mask arm receives the predicted MALUNet
mask. Training data, class-balanced loss, schedule, and seeds are matched.
No test data are loaded or evaluated. Existing checkpoints are read-only.
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
from torch.utils.data import DataLoader

from train_classifier_ablation import CROPS, OUTPUT as ORIGINAL_RESULTS, ClassDataset, Xception_dilation, evaluate


OUTPUT = Path(__file__).resolve().parent / "pilot_mask_optimization"
BASE = ORIGINAL_RESULTS / "rgb" / "best.pth"


class ControlledDataset(ClassDataset):
    def __init__(self, split, mode):
        super().__init__(split, use_mask=True)
        self.mode = mode

    def __getitem__(self, index):
        inputs, label, image_id = super().__getitem__(index)
        if self.mode == "zero":
            inputs[3].zero_()
        return inputs, label, image_id


def class_weights():
    with (CROPS / "train" / "manifest.csv").open(newline="", encoding="utf-8-sig") as file:
        counts = Counter(int(row["label"]) for row in csv.DictReader(file))
    if set(counts) != {0, 1, 2}:
        raise ValueError(f"Unexpected train classes: {counts}")
    raw = torch.tensor([counts[i] ** -0.5 for i in range(3)], dtype=torch.float32)
    return raw / raw.mean(), counts


def initial_model():
    checkpoint = torch.load(BASE, map_location="cpu", weights_only=True)
    rgb_state = checkpoint["model"]
    model = Xception_dilation(input_channel=4, num_classes=3)
    initial = {key: value.clone() for key, value in rgb_state.items()}
    rgb_kernel = rgb_state["conv1.weight"]
    initial["conv1.weight"] = torch.cat([rgb_kernel, torch.zeros_like(rgb_kernel[:, :1])], dim=1)
    model.load_state_dict(initial, strict=True)
    return model.cuda()


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temp.replace(path)


def train_one(mode, steps, eval_every, lr, smoke=False):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = True

    model = initial_model()
    train_data = ControlledDataset("train", mode)
    val_data = ControlledDataset("val", mode)
    weights, counts = class_weights()
    criterion = torch.nn.CrossEntropyLoss(weight=weights.cuda())
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=8, shuffle=False, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda")
    print(f"{mode}: train={len(train_data)} val={len(val_data)} counts={dict(counts)} weights={weights.tolist()}", flush=True)
    iterator = iter(train_loader)
    if smoke:
        inputs, labels, _ = next(iterator)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = criterion(model(inputs.cuda()), labels.cuda())
        scaler.scale(loss).backward()
        print(f"{mode} smoke loss={float(loss):.5f}", flush=True)
        return

    folder = OUTPUT / mode
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "best.pth").exists() or (folder / "status.json").exists():
        raise FileExistsError(f"Pilot output already exists: {folder}; do not overwrite")
    metadata = {
        "mode": mode,
        "split": "official ISIC2017 train 2000 / val 150; test unused",
        "base_checkpoint": str(BASE),
        "steps": steps,
        "eval_every": eval_every,
        "initial_lr": lr,
        "loss": "cross entropy with inverse-sqrt crop-count class weights",
        "class_crop_counts": dict(counts),
        "class_weights": weights.tolist(),
        "mask_policy": "predicted MALUNet binary mask" if mode == "mask" else "zero fourth channel",
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(folder / "metadata.json", metadata)
    initial_validation = evaluate(model, val_loader)
    write_json(folder / "initial_validation.json", initial_validation)
    print(f"{mode} initial val Macro-F1={initial_validation['macro_f1']:.4f}", flush=True)

    best_score = (-1.0, -1.0)
    history = []
    for step in range(1, steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:
            group["lr"] = lr * (1 - (step - 1) / steps) ** 0.9
        accumulated_loss = 0.0
        for _ in range(4):
            try:
                inputs, labels, _ = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                inputs, labels, _ = next(iterator)
            if mode == "zero" and torch.count_nonzero(inputs[:, 3]):
                raise AssertionError("Zero control received nonzero mask values")
            with torch.autocast("cuda", dtype=torch.float16):
                loss = criterion(model(inputs.cuda(non_blocking=True)), labels.cuda(non_blocking=True))
            scaler.scale(loss / 4).backward()
            accumulated_loss += float(loss) / 4
        scaler.step(optimizer)
        scaler.update()

        if step % 50 == 0 or step == 1:
            print(f"{mode} step {step}/{steps} loss={accumulated_loss:.4f}", flush=True)
        if step % eval_every == 0 or step == steps:
            validation = evaluate(model, val_loader)
            recall = validation["classification_report"]["melanoma"]["recall"]
            score = (validation["macro_f1"], recall)
            record = {"step": step, "validation": validation}
            history.append(record)
            if score > best_score:
                best_score = score
                temp = folder / "best.tmp"
                torch.save({"model": model.state_dict(), "step": step, "validation": validation, "metadata": metadata}, temp)
                temp.replace(folder / "best.pth")
            write_json(folder / "history.json", history)
            write_json(folder / "status.json", {
                "stage": "training" if step < steps else "complete",
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "step": step,
                "total_steps": steps,
                "best_macro_f1": best_score[0],
                "best_melanoma_recall": best_score[1],
                "latest_validation": validation,
            })
            print(f"{mode} VAL step={step} Macro-F1={score[0]:.4f} melanoma_recall={recall:.4f}", flush=True)
    del model
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("zero", "mask", "both"), default="both")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable")
    if args.steps <= 0 or args.eval_every <= 0 or args.lr <= 0:
        raise ValueError("Steps, eval interval, and LR must be positive")
    modes = ("zero", "mask") if args.mode == "both" else (args.mode,)
    for mode in modes:
        train_one(mode, args.steps, args.eval_every, args.lr, smoke=args.smoke)


if __name__ == "__main__":
    main()

