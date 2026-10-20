"""Generate MALUNet-predicted masks aligned to each MB-DCNN classification crop.

Only predicted masks are used. No segmentation ground truth is read here.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


MALUNET_CODE = Path(r"D:\Desktop\MALUNet-main\MALUNet-main")
CROPS = Path(r"D:\Desktop\MB-DCNN\dataset\classification_crops")
CHECKPOINT = Path(r"D:\Desktop\MALUNet-ISIC2017\results\malunet_isic17_300_epochs\checkpoints\best-epoch105-loss0.3220.pth")
OUTPUT = Path(r"D:\Desktop\MALUNet-MB-DCNN-ISIC2017\predicted_masks")
sys.path.insert(0, str(MALUNET_CODE))
from models.malunet import MALUNet  # noqa: E402


class CropDataset(Dataset):
    def __init__(self, split, pending):
        self.folder = CROPS / split
        self.names = pending

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        with Image.open(self.folder / name) as image:
            image = image.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.float32)
        # Match MALUNet's per-image min-max normalization used in training.
        low, high = float(array.min()), float(array.max())
        array = (array - low) * (255.0 / max(1.0, high - low))
        return torch.from_numpy(array).permute(2, 0, 1), name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=("train", "val", "test"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0, help="Limit pending files per split for a smoke test")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available")
    model = MALUNet(num_classes=1, input_channels=3, c_list=[8, 16, 24, 32, 48, 64], split_att="fc", bridge=True).cuda()
    model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=True))
    model.eval()

    for split in args.splits:
        folder = CROPS / split
        with (folder / "manifest.csv").open(newline="", encoding="utf-8-sig") as file:
            names = [row["file"] for row in csv.DictReader(file)]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate crop filenames in {split}")
        dest = OUTPUT / split
        dest.mkdir(parents=True, exist_ok=True)
        pending = [name for name in names if not (dest / name).exists()]
        if args.limit:
            pending = pending[: args.limit]
        print(f"{split}: {len(names)} crops, {len(pending)} pending", flush=True)
        loader = DataLoader(CropDataset(split, pending), batch_size=args.batch_size, shuffle=False, num_workers=0)
        done = 0
        with torch.inference_mode():
            for images, batch_names in loader:
                probabilities = model(images.cuda(non_blocking=True).float()).squeeze(1).cpu().numpy()
                for probability, name in zip(probabilities, batch_names):
                    mask = Image.fromarray((probability > 0.5).astype(np.uint8) * 255)
                    mask.resize((224, 224), Image.Resampling.NEAREST).save(dest / name)
                done += len(batch_names)
                if done % 500 < len(batch_names) or done == len(pending):
                    print(f"{split}: {done}/{len(pending)} generated", flush=True)
        if not args.limit:
            missing = [name for name in names if not (dest / name).exists()]
            if missing:
                raise RuntimeError(f"{split}: {len(missing)} masks still missing")
            print(f"{split}: complete, {len(names)} aligned masks", flush=True)


if __name__ == "__main__":
    main()

