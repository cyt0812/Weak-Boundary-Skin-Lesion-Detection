import pandas as pd
from pathlib import Path

csv_path = Path(
    r"D:\GraduationProject\Skin-Lesion-Detection\datasets\ISIC2018\classification\ground_truth\ISIC2018_Task3_Training_GroundTruth\ISIC2018_Task3_Training_GroundTruth.csv"
)

save_path = Path(
    r"D:\GraduationProject\Skin-Lesion-Detection\datasets\ISIC2018\classification\splits\classification_labels.csv"
)

classes = [
    "MEL",
    "NV",
    "BCC",
    "AKIEC",
    "BKL",
    "DF",
    "VASC"
]

df = pd.read_csv(csv_path)

df["label"] = df[classes].values.argmax(axis=1)

df["disease"] = df[classes].idxmax(axis=1)

result = df[["image", "label", "disease"]]

result.to_csv(save_path, index=False)

print("生成完成")
print("数量:", len(result))
print(result.head())