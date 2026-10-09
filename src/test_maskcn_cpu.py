import sys
import torch
import numpy as np
from PIL import Image

# MB-DCNN 项目路径
sys.path.insert(0, r"D:\GraduationProject\MB-DCNN")

from net.models import Xception_dilation


# =========================
# 路径
# =========================
image_path = r"D:\GraduationProject\Skin-Lesion-Detection\datasets\ISIC2018\segmentation\ISIC2018_Task1-2_Training_Input\ISIC_0000000.jpg"

mask_path = r"D:\GraduationProject\Skin-Lesion-Detection\outputs\coarse_masks\train\ISIC_0000000_coarse_mask.png"


# =========================
# CPU
# =========================
device = torch.device("cpu")


print("========== MaskCN CPU 单张测试 ==========")
print("设备：", device)


# =========================
# 创建模型
# =========================
model = Xception_dilation(
    input_channel=4,
    num_classes=3
)

model = model.to(device)
model.eval()

print("模型创建成功")


# =========================
# 读取 RGB 图片
# =========================
image = Image.open(
    image_path
).convert("RGB")

image = image.resize(
    (256, 256),
    Image.Resampling.BILINEAR
)

image_array = np.array(
    image
).astype(np.float32) / 255.0

image_tensor = torch.from_numpy(
    image_array
).permute(2, 0, 1)


# =========================
# 读取 Coarse Mask
# =========================
mask = Image.open(
    mask_path
).convert("L")

mask = mask.resize(
    (256, 256),
    Image.Resampling.NEAREST
)

mask_array = np.array(
    mask
).astype(np.float32) / 255.0

mask_tensor = torch.from_numpy(
    mask_array
).unsqueeze(0)


# =========================
# RGB + Coarse Mask
# =========================
combined = torch.cat(
    [
        image_tensor,
        mask_tensor
    ],
    dim=0
)

print("RGB shape：", image_tensor.shape)
print("Coarse Mask shape：", mask_tensor.shape)
print("MaskCN 输入 shape：", combined.shape)
print("MaskCN 输入 dtype：", combined.dtype)


# =========================
# 添加 Batch 维度
# =========================
input_tensor = combined.unsqueeze(0).to(device)

print("最终输入 shape：", input_tensor.shape)


# =========================
# 前向传播
# =========================
with torch.no_grad():

    output = model(
        input_tensor
    )

print("MaskCN 输出 shape：", output.shape)
print("MaskCN 输出 dtype：", output.dtype)
print("MaskCN 输出：", output)


# =========================
# Softmax
# =========================
probability = torch.softmax(
    output,
    dim=1
)

print("Softmax 概率：", probability)
print("概率总和：", probability.sum(dim=1))


print("========== MaskCN 单张测试完成 ==========")