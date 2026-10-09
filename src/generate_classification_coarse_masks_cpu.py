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

# Task3 分类图片
image_dir = r"D:\GraduationProject\Skin-Lesion-Detection\datasets\ISIC2018\classification\images\ISIC2018_Task3_Training_Input"

# 分割模型
checkpoint_path = r"D:\GraduationProject\Skin-Lesion-Detection\outputs\coarse_sn\coarse_sn_epoch_4.pth"

# 新的输出目录
output_root = r"D:\GraduationProject\Skin-Lesion-Detection\outputs\classification_coarse_masks"


# =========================
# 设置
# =========================

device = torch.device("cpu")

image_size = (256, 256)


os.makedirs(
    output_root,
    exist_ok=True
)


# =========================
# 加载模型
# =========================

print("========== Task3 Coarse Mask 生成 ==========")
print("设备：", device)


model = deeplabv3plus(
    num_classes=2
)


checkpoint = torch.load(
    checkpoint_path,
    map_location=device
)


if "model_state_dict" in checkpoint:
    checkpoint = checkpoint["model_state_dict"]

elif "state_dict" in checkpoint:
    checkpoint = checkpoint["state_dict"]


load_result = model.load_state_dict(
    checkpoint,
    strict=False
)


print(
    "Missing keys：",
    load_result.missing_keys
)

print(
    "Unexpected keys：",
    load_result.unexpected_keys
)


model = model.to(device)

model.eval()


print("模型加载完成")


# =========================
# 图片读取
# =========================

def load_image(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")


    image = image.resize(
        image_size,
        Image.Resampling.BILINEAR
    )


    image_array = (
        np.array(image)
        .astype(np.float32)
        / 255.0
    )


    image_tensor = torch.from_numpy(
        image_array
    ).permute(2,0,1)


    return image_tensor



# =========================
# 生成 mask
# =========================

def generate_coarse_mask(image_tensor):

    image_tensor = (
        image_tensor
        .unsqueeze(0)
        .to(device)
    )


    image_90 = torch.rot90(
        image_tensor,
        1,
        dims=[2,3]
    )

    image_180 = torch.rot90(
        image_tensor,
        2,
        dims=[2,3]
    )

    image_270 = torch.rot90(
        image_tensor,
        3,
        dims=[2,3]
    )


    image_hflip = torch.flip(
        image_tensor,
        dims=[3]
    )


    image_vflip = torch.flip(
        image_tensor,
        dims=[2]
    )


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

        outputs = model(
            tta_images
        )


    probabilities = torch.softmax(
        outputs,
        dim=1
    )[:,1,:,:]


    prob = (
        probabilities[0]
        + torch.rot90(probabilities[1],3,[0,1])
        + torch.rot90(probabilities[2],2,[0,1])
        + torch.rot90(probabilities[3],1,[0,1])
        + torch.flip(probabilities[4],[1])
        + torch.flip(probabilities[5],[0])
    ) / 6.0


    mask = (
        prob >= 0.5
    ).to(torch.uint8) * 255


    return mask.cpu().numpy()



# =========================
# 批量生成
# =========================


images = sorted(
    [
        f for f in os.listdir(image_dir)
        if f.endswith(".jpg")
    ]
)


total = len(images)


print()
print("图片数量：", total)
print()


for index, filename in enumerate(images, start=1):


    name = filename.replace(
        ".jpg",
        ""
    )


    image_path = os.path.join(
        image_dir,
        filename
    )


    output_path = os.path.join(
        output_root,
        name + "_coarse_mask.png"
    )


    if os.path.exists(output_path):
        continue


    image_tensor = load_image(
        image_path
    )


    mask = generate_coarse_mask(
        image_tensor
    )


    Image.fromarray(
        mask
    ).save(
        output_path
    )


    if index == 1 or index % 50 == 0 or index == total:

        ratio = np.mean(
            mask > 0
        )


        print(
            f"{index}/{total} 完成，"
            f"前景比例：{ratio:.4f}"
        )



print()
print("==============================")
print("Task3 Coarse Mask 生成完成")
print(output_root)
print("==============================")