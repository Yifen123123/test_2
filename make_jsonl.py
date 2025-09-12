import json
import re
from pathlib import Path

# 停止條件：遇到這些關鍵詞就不再繼續拼接
STOP_WORDS = ["說明", "依據", "附件", "承辦人", "受文者", "發文日期"]

def extract_title(text: str) -> str:
    lines = text.strip().splitlines()
    capture = False
    collected = []

    for line in lines:
        line = line.strip()
        if not line:
            if capture:
                break
            continue

        # 去掉開頭的「. .. ...」噪音
        line = re.sub(r"^\.{1,}\s*", "", line)

        # 開始條件
        if not capture and re.search(r"(主旨|案由|標題)\s*[:：]", line):
            capture = True
            # 去掉「主旨：」
            line = re.sub(r"^(主旨|案由|標題)\s*[:：]\s*", "", line)
            if line:
                collected.append(line)
            continue

        # 停止條件
        if capture and any(sw in line for sw in STOP_WORDS):
            break

        # 拼接
        if capture:
            collected.append(line)

    return " ".join(collected) if collected else (lines[0] if lines else "")


    # fallback: 如果完全沒抓到，就用第一行
    return " ".join(collected) if collected else (lines[0] if lines else "")

def make_jsonl(input_dir="output_data/train", output_file="train.jsonl"):
    input_dir = Path(input_dir)
    with open(output_file, "w", encoding="utf-8") as f_out:
        for class_dir in input_dir.iterdir():
            if not class_dir.is_dir():
                continue
            label = class_dir.name
            for txt_file in class_dir.glob("*.txt"):
                with open(txt_file, "r", encoding="utf-8", errors="ignore") as f:
                    raw_text = f.read()
                title = extract_title(raw_text)
                record = {"id": txt_file.stem, "title": title, "label": label}
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"已輸出: {output_file}")

if __name__ == "__main__":
    make_jsonl("output_data/train", "train.jsonl")
    make_jsonl("output_data/test", "test.jsonl")
