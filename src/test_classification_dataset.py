import pandas as pd
from pathlib import Path
from PIL import Image
import torch
import numpy as np


# ==========================
# 璺緞
# ==========================

image_root = Path(
    r"D:\GraduationProject\Skin-Lesion-Detection\datasets\ISIC2018\classification\images\ISIC2018_Task3_Training_Input"
)

mask_root = Path(
    r"D:\GraduationProject\Skin-Lesion-Detection\outputs\\classification_coarse_masks"
)

label_csv = Path(
    r"D:\GraduationProject\Skin-Lesion-Detection\datasets\ISIC2018\classification\splits\classification_labels.csv"
)


# ==========================
# 璇诲彇鏍囩
# ==========================

df = pd.read_csv(label_csv)

print("鏍囩鏁伴噺:", len(df))
print(df.head())


# ==========================
# 闅忔満娴嬭瘯绗竴寮?
# ==========================

sample = df.iloc[0]

image_name = sample["image"]
label = sample["label"]
disease = sample["disease"]


# 鍥剧墖璺緞
image_path = image_root / (image_name + ".jpg")

# coarse mask璺緞
mask_path = mask_root / (image_name + "_coarse_mask.png")


print("\n娴嬭瘯鍥剧墖:")
print(image_path)

print("娴嬭瘯Mask:")
print(mask_path)


# ==========================
# 璇诲彇鍥剧墖
# ==========================

image = Image.open(image_path).convert("RGB")
image = image.resize((256, 256))


image_tensor = (
    torch.from_numpy(np.array(image))
    .permute(2, 0, 1)
    .float()
    / 255.0
)


# ==========================
# 璇诲彇Mask
# ==========================

mask = Image.open(mask_path).convert("L")
mask = mask.resize((256, 256))


mask_tensor = (
    torch.from_numpy(np.array(mask))
    .unsqueeze(0)
    .float()
    / 255.0
)


# ==========================
# 鎷兼帴 RGB + Mask
# ==========================

input_tensor = torch.cat(
    [
        image_tensor,
        mask_tensor
    ],
    dim=0
)


print("\n========== Dataset 娴嬭瘯 ==========")

print("Image shape:", image_tensor.shape)

print("Mask shape:", mask_tensor.shape)

print("Fusion input shape:", input_tensor.shape)

print("Input dtype:", input_tensor.dtype)

print("Label:", label)

print("Disease:", disease)

print("Mask unique values:", torch.unique(mask_tensor))

print("===================================")
