import os
import sys
import torch
import numpy as np
from PIL import Image

# MB-DCNN 项目路径
sys.path.insert(0, r"D:\GraduationProject\MB-DCNN")

from net.models import deeplabv3plus


# =========================
# 路径
# =========================
image_dir = r"D:\GraduationProject\Skin-Lesion-Detection\datasets\ISIC2018\segmentation\ISIC2018_Task1-2_Training_Input"

split_dir = r"D:\GraduationProject\Skin-Lesion-Detection\datasets\ISIC2018\splits"

checkpoint_path = r"D:\GraduationProject\Skin-Lesion-Detection\outputs\coarse_sn\coarse_sn_epoch_4.pth"

output_root = r"D:\GraduationProject\Skin-Lesion-Detection\outputs\coarse_masks"


# =========================
# 基本设置
# =========================
device = torch.device("cpu")
image_size = (256, 256)


# =========================
# 创建输出文件夹
# =========================
train_output_dir = os.path.join(output_root, "train")
val_output_dir = os.path.join(output_root, "val")

os.makedirs(train_output_dir, exist_ok=True)
os.makedirs(val_output_dir, exist_ok=True)


# =========================
# 读取模型
# =========================
print("========== Coarse Mask 批量生成 ==========")
print("设备：", device)

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

load_result = model.load_state_dict(checkpoint, strict=False)

print("Missing keys：", load_result.missing_keys)
print("Unexpected keys：", load_result.unexpected_keys)

model = model.to(device)
model.eval()

print("模型加载完成")


# =========================
# 读取图片
# =========================
def load_image(image_path):
    image = Image.open(image_path).convert("RGB")

    original_size = image.size

    image = image.resize(
        image_size,
        Image.Resampling.BILINEAR
    )

    image_array = np.array(image).astype(np.float32) / 255.0

    image_tensor = torch.from_numpy(
        image_array
    ).permute(2, 0, 1)

    return image_tensor, original_size


# =========================
# 生成单张 Coarse Mask
# =========================
def generate_coarse_mask(image_tensor):

    image_tensor = image_tensor.unsqueeze(0).to(device)

    # 6种 TTA
    image_90 = torch.rot90(image_tensor, 1, dims=[2, 3])
    image_180 = torch.rot90(image_tensor, 2, dims=[2, 3])
    image_270 = torch.rot90(image_tensor, 3, dims=[2, 3])
    image_hflip = torch.flip(image_tensor, dims=[3])
    image_vflip = torch.flip(image_tensor, dims=[2])

    tta_images = torch.cat(
        [
            image_tensor,
            image_90,
            image_180,
            image_270,
            image_hflip,
            image_vflip
        ],
        dim=0
    )

    with torch.no_grad():
        outputs = model(tta_images)

    # 取前景概率
    probabilities = torch.softmax(outputs, dim=1)[:, 1, :, :]

    # 恢复 TTA 变换
    prob_original = probabilities[0]

    prob_90 = torch.rot90(
        probabilities[1],
        3,
        dims=[0, 1]
    )

    prob_180 = torch.rot90(
        probabilities[2],
        2,
        dims=[0, 1]
    )

    prob_270 = torch.rot90(
        probabilities[3],
        1,
        dims=[0, 1]
    )

    prob_hflip = torch.flip(
        probabilities[4],
        dims=[1]
    )

    prob_vflip = torch.flip(
        probabilities[5],
        dims=[0]
    )

    # 6个结果平均
    probability = (
        prob_original
        + prob_90
        + prob_180
        + prob_270
        + prob_hflip
        + prob_vflip
    ) / 6.0

    # 二值化
    mask = (probability >= 0.5).to(torch.uint8) * 255

    return mask.cpu().numpy()


# =========================
# 读取 split
# =========================
def read_split(split_file):

    with open(split_file, "r", encoding="utf-8") as f:
        names = [
            line.strip()
            for line in f
            if line.strip()
        ]

    return names


# =========================
# 批量处理
# =========================
def process_split(split_name):

    split_file = os.path.join(
        split_dir,
        split_name + ".txt"
    )

    output_dir = os.path.join(
        output_root,
        split_name
    )

    names = read_split(split_file)

    total = len(names)

    print()
    print("========================================")
    print("开始生成：", split_name)
    print("数量：", total)
    print("========================================")

    for index, name in enumerate(names, start=1):

        image_path = os.path.join(
            image_dir,
            name + ".jpg"
        )

        output_path = os.path.join(
            output_dir,
            name + "_coarse_mask.png"
        )

        # 如果已经存在，就跳过
        if os.path.exists(output_path):
            continue

        image_tensor, original_size = load_image(
            image_path
        )

        mask = generate_coarse_mask(
            image_tensor
        )

        Image.fromarray(mask).save(
            output_path
        )

        if index == 1 or index % 50 == 0 or index == total:

            foreground_ratio = np.mean(
                mask > 0
            )

            print(
                f"{split_name}: "
                f"{index}/{total} "
                f"完成，"
                f"前景比例："
                f"{foreground_ratio:.4f}"
            )


# =========================
# 开始生成
# =========================
process_split("train")

process_split("val")


print()
print("========================================")
print("全部 Coarse Mask 生成完成")
print("输出目录：")
print(output_root)
print("========================================")