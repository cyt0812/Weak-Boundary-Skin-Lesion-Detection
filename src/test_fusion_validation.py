import sys
import time
import torch

sys.path.insert(0, r"D:\GraduationProject\Skin-Lesion-Detection\src")

from classification_model_fusion import SkinLesionFeatureFusionClassifier
from classification_dataset import SkinLesionClassificationDataset
from torch.utils.data import DataLoader


image_dir = r"D:\GraduationProject\Skin-Lesion-Detection\datasets\ISIC2018\classification\images\ISIC2018_Task3_Training_Input"
mask_dir = r"D:\GraduationProject\Skin-Lesion-Detection\outputs\classification_coarse_masks"
label_csv = r"D:\GraduationProject\Skin-Lesion-Detection\datasets\ISIC2018\classification\splits\classification_labels.csv"
split_file = r"D:\GraduationProject\Skin-Lesion-Detection\datasets\ISIC2018\classification\splits\val_cls.txt"


print("========== Feature Fusion 验证测试 ==========")

dataset = SkinLesionClassificationDataset(
    image_dir,
    mask_dir,
    label_csv,
    split_file,
    image_size=(128, 128)
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0
)

model = SkinLesionFeatureFusionClassifier(num_classes=7)
model.eval()

print("验证集数量:", len(dataset))
print("验证批次:", len(loader))
print("模型参数量:", sum(p.numel() for p in model.parameters()))

start_time = time.time()
total = 0
correct = 0

with torch.no_grad():
    for batch_index, batch in enumerate(loader, start=1):

        logits = model(
            batch["image"],
            batch["mask"]
        )

        predictions = logits.argmax(dim=1)

        correct += (predictions == batch["label"]).sum().item()
        total += batch["label"].size(0)

        if batch_index % 50 == 0:
            elapsed = time.time() - start_time
            print(
                f"已完成: {batch_index}/{len(loader)} "
                f"| 当前耗时: {elapsed:.2f} 秒"
            )

elapsed = time.time() - start_time

print()
print("========== 测试完成 ==========")
print("总批次:", len(loader))
print("总图片:", total)
print("正确数量:", correct)
print("简单Accuracy:", round(correct / total, 4))
print("总耗时:", round(elapsed, 2), "秒")