
import pandas as pd
from pathlib import Path

# 自动定位项目根目录，避免依赖固定的绝对路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent

CSV_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "ISIC2018"
    / "classification"
    / "ground_truth"
    / "ISIC2018_Task3_Training_GroundTruth"
    / "ISIC2018_Task3_Training_GroundTruth.csv"
)

SAVE_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "ISIC2018"
    / "classification"
    / "splits"
    / "classification_labels.csv"
)

CLASSES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]


def main():
    if not CSV_PATH.is_file():
        raise FileNotFoundError(f"找不到原始标签 CSV：{CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    required_columns = ["image"] + CLASSES
    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(f"原始 CSV 缺少列：{missing_columns}")

    if df["image"].isna().any():
        raise ValueError("图片名称中存在空值")

    if df["image"].duplicated().any():
        raise ValueError("原始 CSV 中存在重复图片名称")

    labels = df[CLASSES].apply(pd.to_numeric, errors="coerce")

    if labels.isna().any().any():
        raise ValueError("类别标签中存在空值或非数字内容")

    # 每个类别只能是 0 或 1
    valid_values = labels.isin([0, 1]).all().all()
    if not valid_values:
        raise ValueError("类别标签中存在 0 和 1 以外的数值")

    # 每张图片必须且只能属于一个类别
    positive_counts = labels.sum(axis=1)
    invalid_rows = positive_counts != 1

    if invalid_rows.any():
        print("发现无效标签行数：", int(invalid_rows.sum()))
        print(df.loc[invalid_rows, ["image"] + CLASSES].head(10))
        raise ValueError("每张图片必须恰好有一个类别标签为 1")

    result = pd.DataFrame({
        "image": df["image"],
        "label": labels.values.argmax(axis=1),
        "disease": labels.idxmax(axis=1),
    })

    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(SAVE_PATH, index=False, encoding="utf-8")

    print("标签生成成功")
    print("图片数量：", len(result))
    print("类别编号：")
    for index, disease in enumerate(CLASSES):
        count = int((result["disease"] == disease).sum())
        print(f"  {index}: {disease} = {count}")
    print("保存位置：", SAVE_PATH)
    print("\n前 5 行：")
    print(result.head().to_string(index=False))


if __name__ == "__main__":
    main()
