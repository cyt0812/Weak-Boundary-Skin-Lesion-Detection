import os
import pandas as pd
from sklearn.model_selection import train_test_split


# =========================
# 路径
# =========================

label_csv = r"D:\GraduationProject\Skin-Lesion-Detection\datasets\ISIC2018\classification\splits\classification_labels.csv"

output_dir = r"D:\GraduationProject\Skin-Lesion-Detection\datasets\ISIC2018\classification\splits"


# =========================
# 读取标签
# =========================

df = pd.read_csv(label_csv)

print("总数量:", len(df))


# =========================
# 第一次划分
# train 70%
# temp 30%
# =========================

train_df, temp_df = train_test_split(
    df,
    test_size=0.3,
    random_state=42,
    stratify=df["disease"]
)


# =========================
# 第二次划分
# val 15%
# test 15%
# =========================

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=42,
    stratify=temp_df["disease"]
)


print("Train:", len(train_df))
print("Val:", len(val_df))
print("Test:", len(test_df))


# =========================
# 保存 txt
# =========================

def save_txt(data, name):

    path = os.path.join(
        output_dir,
        name
    )

    with open(path, "w", encoding="utf-8") as f:

        for image in data["image"]:
            f.write(image + "\n")

    print("保存:", path)



save_txt(train_df, "train_cls.txt")
save_txt(val_df, "val_cls.txt")
save_txt(test_df, "test_cls.txt")


print("划分完成")