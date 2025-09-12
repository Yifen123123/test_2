import os
import shutil
import random
from pathlib import Path

def split_data(input_dir="data", output_dir="output_data", test_ratio=0.2, min_test=3, seed=42):
    random.seed(seed)
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # 清理並建立新的輸出資料夾
    if output_dir.exists():
        shutil.rmtree(output_dir)
    (output_dir / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "test").mkdir(parents=True, exist_ok=True)

    # 逐類別處理
    for class_dir in input_dir.iterdir():
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        files = list(class_dir.glob("*"))  # 可改成 .glob("*.txt") 限制檔案型別
        if not files:
            continue

        # 打亂
        random.shuffle(files)

        # 計算 test 集數量
        n_total = len(files)
        n_test = max(min_test, int(n_total * test_ratio))
        n_test = min(n_test, n_total - 1)  # 保證至少留一份在 train

        test_files = files[:n_test]
        train_files = files[n_test:]

        # 建立輸出子資料夾
        train_out = output_dir / "train" / class_name
        test_out = output_dir / "test" / class_name
        train_out.mkdir(parents=True, exist_ok=True)
        test_out.mkdir(parents=True, exist_ok=True)

        # 複製檔案
        for f in train_files:
            shutil.copy(f, train_out / f.name)
        for f in test_files:
            shutil.copy(f, test_out / f.name)

        print(f"[{class_name}] total={n_total}, train={len(train_files)}, test={len(test_files)}")

if __name__ == "__main__":
    split_data("data", "output_data", test_ratio=0.2, min_test=3, seed=42)
