"""ISIC2017 MB-DCNN RGB vs RGB+MALUNet-mask controlled comparison.

Both arms use the same official split, crop manifests, architecture family,
pretraining, seed, augmentation, training schedule, and image-level evaluation.
The test set is evaluated only after selecting a checkpoint on validation data.
"""

import argparse
import csv
import ctypes
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


CODE = Path(r"D:\Desktop\MB-DCNN-ISIC2017-Reproduction")
CROPS = Path(r"D:\Desktop\MB-DCNN\dataset\classification_crops")
MASKS = Path(r"D:\Desktop\MALUNet-MB-DCNN-ISIC2017\predicted_masks")
PRETRAINED = Path(r"D:\Desktop\MB-DCNN\models\xception-43020ad28.pth")
OUTPUT = Path(r"D:\Desktop\MALUNet-MB-DCNN-ISIC2017\results")
CLASS_NAMES = ["nevus", "melanoma", "seborrheic_keratosis"]
sys.path.insert(0, str(CODE))
from net.models import Xception_dilation  # noqa: E402


class ClassDataset(Dataset):
    def __init__(self, split, use_mask):
        self.split = split
        self.use_mask = use_mask
        self.crop_dir = CROPS / split
        self.mask_dir = MASKS / split
        with (self.crop_dir / "manifest.csv").open(newline="", encoding="utf-8-sig") as file:
            self.rows = list(csv.DictReader(file))
        if use_mask:
            missing = [r["file"] for r in self.rows if not (self.mask_dir / r["file"]).exists()]
            if missing:
                raise RuntimeError(f"{split}: {len(missing)} predicted MALUNet masks missing")
        self.augment = transforms.Compose([
            transforms.RandomAffine(10, translate=(0, 0.1), scale=(0.9, 1.1), shear=5.729578),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.crop_dir / row["file"]) as source:
            image = source.convert("RGB")
        mask = None
        if self.use_mask:
            with Image.open(self.mask_dir / row["file"]) as source:
                mask = source.convert("L")
        if self.split == "train":
            rng = torch.get_rng_state()
            image = self.augment(image)
            if mask is not None:
                torch.set_rng_state(rng)
                mask = self.augment(mask)
        rgb = torch.from_numpy(np.asarray(image, dtype=np.float32).copy() / 255.0).permute(2, 0, 1)
        if mask is not None:
            channel = torch.from_numpy((np.asarray(mask) > 0).astype(np.float32)).unsqueeze(0)
            rgb = torch.cat([rgb, channel], dim=0)
        return rgb, int(row["label"]), row["image_id"]


def load_pretrained(model, use_mask):
    raw = torch.load(PRETRAINED, map_location="cpu", weights_only=True)
    state = {key.removeprefix("module."): value for key, value in raw.items()}
    if use_mask:
        rgb_weight = state["conv1.weight"]
        state["conv1.weight"] = torch.cat([rgb_weight, rgb_weight.mean(1, keepdim=True)], dim=1)
    own = model.state_dict()
    matched = {key: value for key, value in state.items() if key in own and own[key].shape == value.shape}
    if len(matched) < 100:
        raise RuntimeError(f"Only {len(matched)} pretrained tensors matched")
    model.load_state_dict(matched, strict=False)
    print(f"Pretrained tensors matched: {len(matched)}/{len(own)}", flush=True)


def evaluate(model, loader, with_predictions=False):
    model.eval()
    scores = {}
    labels = {}
    with torch.inference_mode():
        for images, targets, ids in loader:
            probs = model(images.cuda(non_blocking=True)).softmax(1).cpu().numpy()
            for isic_id, target, probability in zip(ids, targets.tolist(), probs):
                scores.setdefault(isic_id, []).append(probability)
                labels[isic_id] = target
    ids = list(scores)
    y = np.array([labels[isic_id] for isic_id in ids])
    probability = np.array([np.mean(scores[isic_id], axis=0) for isic_id in ids])
    predicted = probability.argmax(1)
    aucs = [float(roc_auc_score(y == category, probability[:, category])) for category in range(3)]
    metrics = {
        "images": len(ids),
        "accuracy": float(accuracy_score(y, predicted)),
        "macro_f1": float(f1_score(y, predicted, labels=[0, 1, 2], average="macro", zero_division=0)),
        "auc_per_class": aucs,
        "selection_auc_mean_melanoma_sk": float(np.mean(aucs[1:])),
        "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1, 2]).tolist(),
        "classification_report": classification_report(
            y, predicted, labels=[0, 1, 2], target_names=CLASS_NAMES, output_dict=True, zero_division=0
        ),
    }
    if with_predictions:
        return metrics, [(isic_id, int(truth), int(pred), *map(float, probs)) for isic_id, truth, pred, probs in zip(ids, y, predicted, probability)]
    return metrics


def atomic_save(value, path):
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def write_status(folder, stage, **details):
    status = {"stage": stage, "time": time.strftime("%Y-%m-%d %H:%M:%S"), **details}
    temporary = folder / "status.tmp"
    temporary.write_text(json.dumps(status, indent=2), encoding="utf-8")
    temporary.replace(folder / "status.json")
    print(json.dumps(status), flush=True)


def train(mode, steps, smoke=False):
    use_mask = mode == "malunet_mask"
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = True
    model = Xception_dilation(input_channel=4 if use_mask else 3, num_classes=3).cuda()
    load_pretrained(model, use_mask)
    train_set = ClassDataset("train", use_mask)
    val_set = ClassDataset("val", use_mask)
    test_set = ClassDataset("test", use_mask)
    print(f"{mode}: crops train={len(train_set)}, val={len(val_set)}, test={len(test_set)}", flush=True)
    if smoke:
        images, targets, _ = next(iter(DataLoader(train_set, batch_size=4, num_workers=0)))
        with torch.autocast("cuda", dtype=torch.float16):
            loss = torch.nn.functional.cross_entropy(model(images.cuda()), targets.cuda())
        loss.backward()
        print(f"{mode}: smoke loss={float(loss):.5f}, GPU peak GiB={torch.cuda.max_memory_allocated()/1024**3:.2f}", flush=True)
        return

    folder = OUTPUT / mode
    folder.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    latest = folder / "latest.pth"
    best_path = folder / "best.pth"
    start, best = 0, -1.0
    if latest.exists():
        saved = torch.load(latest, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        start, best = saved["step"], saved["best"]
        random.setstate(saved["random_state"])
        np.random.set_state(saved["numpy_state"])
        torch.set_rng_state(saved["torch_state"])
        torch.cuda.set_rng_state_all(saved["cuda_state"])
        print(f"{mode}: resumed at step {start}", flush=True)

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False, num_workers=0)
    iterator = iter(train_loader)
    write_status(folder, "training", step=start, total_steps=steps, best_validation_auc=best)
    for step in range(start, steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:
            group["lr"] = 1e-4 * (1 - step / steps) ** 0.9
        loss_sum = 0.0
        for _ in range(4):
            try:
                images, targets, _ = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                images, targets, _ = next(iterator)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = torch.nn.functional.cross_entropy(model(images.cuda(non_blocking=True)), targets.cuda(non_blocking=True))
            scaler.scale(loss / 4).backward()
            loss_sum += float(loss) / 4
        scaler.step(optimizer)
        scaler.update()
        if step % 20 == 0:
            print(f"{mode} step {step + 1}/{steps} loss={loss_sum:.5f}", flush=True)
        if (step > 500 and step % 100 == 0) or step + 1 == steps:
            metrics = evaluate(model, val_loader)
            score = metrics["selection_auc_mean_melanoma_sk"]
            if score > best:
                best = score
                atomic_save({"model": model.state_dict(), "step": step + 1, "validation": metrics}, best_path)
            atomic_save({
                "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                "step": step + 1, "best": best, "random_state": random.getstate(),
                "numpy_state": np.random.get_state(), "torch_state": torch.get_rng_state(),
                "cuda_state": torch.cuda.get_rng_state_all(),
            }, latest)
            write_status(folder, "training", step=step + 1, total_steps=steps, validation=metrics, best_validation_auc=best)

    model.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=True)["model"])
    results, predictions = evaluate(model, DataLoader(test_set, batch_size=8, shuffle=False, num_workers=0), with_predictions=True)
    results.update({"mode": mode, "steps": steps, "split": "official ISIC2017 2000/150/600", "mask_source": "MALUNet ISIC2017 best epoch 105" if use_mask else "none"})
    (folder / "test_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    with (folder / "test_predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image_id", "true_label", "predicted_label", "prob_nevus", "prob_melanoma", "prob_seborrheic_keratosis"])
        writer.writerows(predictions)
    write_status(folder, "complete", results=results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("rgb", "malunet_mask", "both"), required=True)
    parser.add_argument("--steps", type=int, default=50001)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is unavailable")
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        modes = ("rgb", "malunet_mask") if args.mode == "both" else (args.mode,)
        for mode in modes:
            train(mode, args.steps, args.smoke)
            torch.cuda.empty_cache()
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == "__main__":
    main()

