
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "ISIC2018"
    / "classification"
)

IMAGE_DIR = DATA_ROOT / "images" / "ISIC2018_Task3_Training_Input"
SPLIT_DIR = DATA_ROOT / "splits"
LABEL_CSV = SPLIT_DIR / "classification_labels.csv"

CLASS_NAMES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
RANDOM_STATE = 42


def main():
    if not LABEL_CSV.is_file():
        raise FileNotFoundError(
            f"找不到标签文件：{LABEL_CSV}\n"
            "请先运行 generate_classification_labels.py"
        )

    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(f"找不到图片目录：{IMAGE_DIR}")

    df = pd.read_csv(LABEL_CSV)

    required_columns = {"image", "label", "disease"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"标签 CSV 必须包含列：{required_columns}")

    if df["image"].isna().any() or df["image"].duplicated().any():
        raise ValueError("标签文件存在空图片名称或重复图片名称")

    if df["disease"].isna().any():
        raise ValueError("标签文件存在空类别")

    if set(df["disease"].unique()) != set(CLASS_NAMES):
        raise ValueError("标签类别与预期的七个类别不一致")

    # 检查类别名称和编号是否一一对应
    expected_labels = {
        name: index for index, name in enumerate(CLASS_NAMES)
    }
    actual_labels = df.groupby("disease")["label"].unique()

    for name, expected_index in expected_labels.items():
        values = actual_labels[name]
        if len(values) != 1 or int(values[0]) != expected_index:
            raise ValueError(f"类别 {name} 的标签编号不正确")

    # 检查每个标签名称都能找到对应图片
    missing_images = [
        name for name in df["image"]
        if not (IMAGE_DIR / f"{name}.jpg").is_file()
    ]

    if missing_images:
        print("找不到的图片数量：", len(missing_images))
        print("示例：", missing_images[:10])
        raise FileNotFoundError("部分标签无法对应实际图片")

    # 第一次：70% 训练集，30% 临时集合
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=df["disease"],
    )

    # 第二次：将临时集合均分为验证集和测试集
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temp_df["disease"],
    )

    splits = {
        "train_cls.txt": train_df,
        "val_cls.txt": val_df,
        "test_cls.txt": test_df,
    }

    # 检查划分互不重叠
    train_names = set(train_df["image"])
    val_names = set(val_df["image"])
    test_names = set(test_df["image"])

    if train_names & val_names:
        raise ValueError("训练集与验证集存在重叠")
    if train_names & test_names:
        raise ValueError("训练集与测试集存在重叠")
    if val_names & test_names:
        raise ValueError("验证集与测试集存在重叠")

    all_names = train_names | val_names | test_names
    if len(all_names) != len(df):
        raise ValueError("划分后图片数量与原始标签数量不一致")

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    for filename, subset in splits.items():
        output_path = SPLIT_DIR / filename
        subset["image"].to_csv(
            output_path,
            index=False,
            header=False,
            encoding="utf-8",
        )
        print(f"{filename}: {len(subset)} 张")

    print("\n各子集类别分布：")
    for split_name, subset in [
        ("训练集", train_df),
        ("验证集", val_df),
        ("测试集", test_df),
    ]:
        print(f"\n{split_name}（{len(subset)} 张）")
        counts = (
            subset["disease"]
            .value_counts()
            .reindex(CLASS_NAMES, fill_value=0)
        )
        for name, count in counts.items():
            print(f"  {name}: {count}")

    print("\n数据划分验证通过")
    print("训练集、验证集、测试集没有重叠")
    print("所有图片均有标签且能找到对应文件")
    print("随机种子：", RANDOM_STATE)
    print("划分目录：", SPLIT_DIR)


if __name__ == "__main__":
    main()
