import os
import torch
import numpy as np
from PIL import Image
import sys

sys.path.insert(0, r"D:\GraduationProject\MB-DCNN")
from net.models import deeplabv3plus


# =========================
# 1. 路径
# =========================
image_path = r"D:\GraduationProject\Skin-Lesion-Detection\datasets\ISIC2018\segmentation\ISIC2018_Task1-2_Training_Input\ISIC_0000000.jpg"

checkpoint_path = r"D:\GraduationProject\Skin-Lesion-Detection\outputs\coarse_sn\coarse_sn_epoch_1.pth"

output_dir = r"D:\GraduationProject\Skin-Lesion-Detection\outputs\coarse_masks_test"

os.makedirs(output_dir, exist_ok=True)


# =========================
# 2. CPU
# =========================
device = torch.device("cpu")

print("========== Coarse Mask 单张测试 ==========")
print("设备：", device)


# =========================
# 3. 创建模型
# =========================
model = deeplabv3plus(num_classes=2)

checkpoint = torch.load(
    checkpoint_path,
    map_location=device
)

# 我们的训练脚本保存的是完整 checkpoint
if "model_state_dict" in checkpoint:
    checkpoint = checkpoint["model_state_dict"]

# 如果是直接保存 state_dict 的情况，也兼容
elif "state_dict" in checkpoint:
    checkpoint = checkpoint["state_dict"]

# 如果 checkpoint 中有 module. 前缀，去掉
new_checkpoint = {}

for key, value in checkpoint.items():
    if key.startswith("module."):
        key = key[7:]
    new_checkpoint[key] = value

missing_keys, unexpected_keys = model.load_state_dict(
    new_checkpoint,
    strict=False
)

print("模型加载完成")
print("Missing keys：", missing_keys)
print("Unexpected keys：", unexpected_keys)

model = model.to(device)

# 推理模式
model.eval()


# =========================
# 4. 读取图片
# =========================
image = Image.open(image_path).convert("RGB")

print("原始图片尺寸：", image.size)

# CoarseSN 使用 256 × 256
image = image.resize(
    (256, 256),
    Image.Resampling.BILINEAR
)

image_np = np.array(image).astype(np.float32) / 255.0

# HWC → CHW
image_tensor = torch.from_numpy(
    image_np
).permute(2, 0, 1)

# 增加 batch 维度
image_tensor = image_tensor.unsqueeze(0)

print("输入 Tensor shape：", image_tensor.shape)
print("输入 Tensor dtype：", image_tensor.dtype)


# =========================
# 5. 六种 TTA
# =========================
image = image_tensor.to(device)

rot_90 = torch.rot90(image, 1, [2, 3])
rot_180 = torch.rot90(image, 2, [2, 3])
rot_270 = torch.rot90(image, 3, [2, 3])

hor_flip = torch.flip(image, [-1])
ver_flip = torch.flip(image, [-2])

images = torch.cat(
    [
        image,
        rot_90,
        rot_180,
        rot_270,
        hor_flip,
        ver_flip
    ],
    dim=0
)

print("TTA 输入 shape：", images.shape)


# =========================
# 6. CoarseSN 推理
# =========================
with torch.no_grad():

    predictions = model(images)

print("模型输出 shape：", predictions.shape)


# =========================
# 7. 将 TTA 结果恢复到原方向
# =========================
pred = (
    predictions[0:1]
    + torch.rot90(predictions[1:2], 3, [2, 3])
    + torch.rot90(predictions[2:3], 2, [2, 3])
    + torch.rot90(predictions[3:4], 1, [2, 3])
    + torch.flip(predictions[4:5], [-1])
    + torch.flip(predictions[5:6], [-2])
)


# =========================
# 8. Softmax + Argmax
# =========================
pred = torch.softmax(pred, dim=1)

pred_class = torch.argmax(
    pred,
    dim=1
)

# [1, 256, 256] → [256, 256]
mask = pred_class[0].cpu().numpy()


# =========================
# 9. 转成 0 / 255
# =========================
mask = (mask * 255).astype(np.uint8)


# =========================
# 10. 保存
# =========================
output_path = os.path.join(
    output_dir,
    "ISIC_0000000_coarse_mask.png"
)

Image.fromarray(mask).save(output_path)


# =========================
# 11. 输出检查
# =========================
print("Coarse Mask shape：", mask.shape)
print("Mask 数据类型：", mask.dtype)
print("Mask 唯一值：", np.unique(mask))
print("保存路径：", output_path)

print("========== 单张测试完成 ==========")