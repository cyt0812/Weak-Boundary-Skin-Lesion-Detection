"""Prepare the official ISIC2017 train/val segmentation pairs for MALUNet.

The source dataset is read-only. Output filenames retain the official ISIC ID.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image


SPLITS = {
    "train": ("ISIC-2017_Training_Data", "ISIC-2017_Training_Part1_GroundTruth"),
    "val": ("ISIC-2017_Validation_Data", "ISIC-2017_Validation_Part1_GroundTruth"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    split_ids = {}
    for split, (image_dir, mask_dir) in SPLITS.items():
        source_images = args.raw / split / image_dir
        source_masks = args.raw / split / mask_dir
        images = sorted(source_images.glob("ISIC_*.jpg"))
        ids = {path.stem for path in images}
        masks = {path.name.removesuffix("_segmentation.png"): path for path in source_masks.glob("ISIC_*_segmentation.png")}
        if not images or ids != set(masks):
            raise ValueError(f"{split}: image/mask IDs differ; images={len(ids)}, masks={len(masks)}")
        split_ids[split] = ids
        out_images = args.output / split / "images"
        out_masks = args.output / split / "masks"
        out_images.mkdir(parents=True, exist_ok=True)
        out_masks.mkdir(parents=True, exist_ok=True)

        for number, source_image in enumerate(images, 1):
            isic_id = source_image.stem
            source_mask = masks[isic_id]
            with Image.open(source_image) as image, Image.open(source_mask) as mask:
                if image.size != mask.size:
                    raise ValueError(f"{isic_id}: image/mask dimensions differ")
                rgb = image.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
                binary = np.asarray(mask.convert("L").resize((256, 256), Image.Resampling.NEAREST)) > 127
                rgb.save(out_images / f"{isic_id}.png")
                Image.fromarray(binary.astype(np.uint8) * 255).save(out_masks / f"{isic_id}.png")
            rows.append({"split": split, "isic_id": isic_id, "source_image": str(source_image), "source_mask": str(source_mask)})
            if number % 250 == 0 or number == len(images):
                print(f"{split}: {number}/{len(images)}", flush=True)

    if split_ids["train"] & split_ids["val"]:
        raise ValueError("Training and validation ID overlap")
    manifest = args.output / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Prepared {len(rows)} pairs; manifest: {manifest}")


if __name__ == "__main__":
    main()

