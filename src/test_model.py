import sys
import numpy as np
import torch


# 添加 MB-DCNN 项目路径
sys.path.insert(0, r"D:\GraduationProject\MB-DCNN")

from net.models import Xception_dilation


print("正在创建 MaskCN 模型...")

model = Xception_dilation(
    input_channel=4,
    num_classes=3
)

model.eval()

print("MaskCN 模型创建成功！")


# 创建假的 4 通道输入
# RGB + Coarse Mask
input_data = torch.randn(1, 4, 256, 256)

print("Input shape：", input_data.shape)


# =========================
# MaskCN 前向传播
# =========================

with torch.no_grad():
    preds = model(input_data)

print("Classification output shape：", preds.shape)


# =========================
# 获取预测类别
# =========================

class_idx = preds.argmax(dim=1)

print("Predicted class：", class_idx.item())


# =========================
# 获取 MaskCN 中间特征
# =========================

model_layers = model.get_layers()

feature = model_layers[0]

print("Feature shape：", feature.shape)


# =========================
# 获取分类层权重
# =========================

params = list(model.parameters())

weights = np.squeeze(
    params[-2].data.cpu().numpy()
)

print("Classification weights shape：", weights.shape)


# =========================
# 生成 CAM
# =========================

bz, nc, h, w = feature.shape

output_cam = []

for idx in range(bz):

    cam = np.zeros(
        (h, w),
        dtype=np.float32
    )

    for i, weight in enumerate(
        weights[class_idx[idx]]
    ):
        cam += (
            weight
            * feature[idx][i].data.cpu().numpy()
        )

    cam_img = np.maximum(cam, 0)

    cam_img = cam / np.max(cam_img)

    output_cam.append(cam_img)


# =========================
# 转换成 Tensor
# =========================

cam = torch.from_numpy(
    np.stack(output_cam)
).unsqueeze(1)


print("CAM shape：", cam.shape)
print("CAM 数据类型：", cam.dtype)
print("CAM 最小值：", cam.min().item())
print("CAM 最大值：", cam.max().item())