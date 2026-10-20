"""Generate train-crop OOF masks from completed five-fold MALUNet models.

Default is a read-only audit. --execute requires all five finished checkpoints
and writes only a new OOF directory, never the existing predicted masks.
"""

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from prepare_oof_mask_folds import OUTPUT as PLAN_ROOT
from run_malunet_isic2017 import MALUNET_PROJECT, RESULTS_ROOT
from train_classifier_ablation import CROPS


sys.path.insert(0, str(MALUNET_PROJECT))
from models.malunet import MALUNet  # noqa: E402


ASSIGNMENTS = PLAN_ROOT / "fold_assignments.csv"
TRAIN_MANIFEST = CROPS / "train" / "manifest.csv"
OUTPUT = Path(r"D:\Desktop\MALUNet-MB-DCNN-ISIC2017\predicted_masks_oof5_train")


class CropDataset(torch.utils.data.Dataset):
    def __init__(self, names):
        self.names = names

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        with Image.open(CROPS / "train" / name) as source:
            image = source.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.float32)
        low, high = float(array.min()), float(array.max())
        array = (array - low) * (255.0 / max(1.0, high - low))
        return torch.from_numpy(array).permute(2, 0, 1), name


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_plan():
    with ASSIGNMENTS.open(newline="", encoding="utf-8") as file:
        fold_by_id = {row["image_id"]: int(row["held_out_fold"]) for row in csv.DictReader(file)}
    if len(fold_by_id) != 2000 or set(fold_by_id.values()) != set(range(5)):
        raise ValueError("Expected 2000 IDs assigned to folds 0-4")
    by_fold = {fold: [] for fold in range(5)}
    with TRAIN_MANIFEST.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != 21768 or len({row["file"] for row in rows}) != len(rows):
        raise ValueError("Expected 21,768 unique classification-train crops")
    if {row["image_id"] for row in rows} != set(fold_by_id):
        raise ValueError("Classification crop IDs differ from fold manifest")
    for row in rows:
        by_fold[fold_by_id[row["image_id"]]].append(row["file"])
    if sum(len(names) for names in by_fold.values()) != 21768:
        raise ValueError("Fold crops are incomplete")
    return by_fold


def checkpoint_for_fold(fold, epochs):
    directory = RESULTS_ROOT / f"oof5_fold{fold}_epochs{epochs}" / "checkpoints"
    checkpoints = list(directory.glob("best-epoch*-loss*.pth"))
    if len(checkpoints) != 1:
        raise FileNotFoundError(f"Fold {fold}: expected one finished best checkpoint in {directory}")
    return checkpoints[0]


def model_from_checkpoint(path):
    model = MALUNet(num_classes=1, input_channels=3, c_list=[8, 16, 24, 32, 48, 64], split_att="fc", bridge=True).cuda()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, choices=(105, 300), default=105)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("Batch size must be positive")
    by_fold = read_plan()
    print("OOF train-crop counts by held-out fold:", dict(Counter({fold: len(names) for fold, names in by_fold.items()})))
    print(f"total={sum(map(len, by_fold.values()))}, destination={OUTPUT}")
    if not args.execute:
        print("Dry run only. No models loaded and no masks written.")
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable")
    checkpoints = {fold: checkpoint_for_fold(fold, args.epochs) for fold in range(5)}
    metadata = {
        "fold_manifest_sha256": file_hash(ASSIGNMENTS),
        "crop_manifest_sha256": file_hash(TRAIN_MANIFEST),
        "checkpoints": {str(fold): {"path": str(path), "sha256": file_hash(path)} for fold, path in checkpoints.items()},
        "epochs": args.epochs,
        "expected_crops": 21768,
        "purpose": "OOF predicted masks for ISIC2017 classification training only",
    }
    metadata_path = OUTPUT / "metadata.json"
    if OUTPUT.exists():
        if not metadata_path.is_file() or json.loads(metadata_path.read_text(encoding="utf-8")) != metadata:
            raise FileExistsError(f"Existing destination has different or missing metadata: {OUTPUT}")
    else:
        OUTPUT.mkdir(parents=True)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    for fold, names in by_fold.items():
        pending = [name for name in names if not (OUTPUT / name).exists()]
        print(f"fold {fold}: {len(pending)}/{len(names)} pending", flush=True)
        if not pending:
            continue
        model = model_from_checkpoint(checkpoints[fold])
        loader = DataLoader(CropDataset(pending), batch_size=args.batch_size, shuffle=False, num_workers=0)
        done = 0
        with torch.inference_mode():
            for images, batch_names in loader:
                probabilities = model(images.cuda(non_blocking=True).float()).squeeze(1).cpu().numpy()
                for probability, name in zip(probabilities, batch_names):
                    mask = Image.fromarray((probability > 0.5).astype(np.uint8) * 255)
                    mask.resize((224, 224), Image.Resampling.NEAREST).save(OUTPUT / name)
                done += len(batch_names)
                if done % 500 < len(batch_names) or done == len(pending):
                    print(f"fold {fold}: {done}/{len(pending)} generated", flush=True)
        del model
        torch.cuda.empty_cache()
    missing = [name for names in by_fold.values() for name in names if not (OUTPUT / name).is_file()]
    if missing:
        raise RuntimeError(f"OOF generation incomplete: {len(missing)} files missing")
    print("OOF train masks complete: 21,768 aligned crop masks", flush=True)


if __name__ == "__main__":
    main()

