import json
import re
from pathlib import Path

# 停止條件：遇到這些關鍵詞就不再繼續拼接
STOP_WORDS = ["說明", "依據", "附件", "承辦人", "受文者", "發文日期"]

import re

# 行首才算欄位標題的停止條件
STOP_HEADINGS = r"(說明|理由)"
STOP_RE = re.compile(rf"^\s*{STOP_HEADINGS}\s*[:：]?\s*$|^\s*{STOP_HEADINGS}\s*[:：]", re.UNICODE)
START_RE = re.compile(r"^(主旨|主文)\s*[:：]?\s*", re.UNICODE)

def _clean_line(line: str) -> str:
    line = line.strip()
    # 去掉行首 . .. ... 之類
    line = re.sub(r"^\.{1,}\s*", "", line)
    # 全形空白→半形空白，壓縮多重空白
    line = line.replace("\u3000", " ")
    line = re.sub(r"\s+", " ", line)
    return line

def extract_title(text: str) -> str:
    lines = text.splitlines()
    capture = False
    collected = []

    for raw in lines:
        line = _clean_line(raw)
        if not line:
            if capture:  # 已開始抓，遇空行就結束
                break
            continue

        if not capture:
            # 行首出現 主旨/案由/標題（冒號可有可無）就開始擷取
            if START_RE.search(line):
                capture = True
                line = START_RE.sub("", line)  # 去掉「主旨：」
                if line:
                    collected.append(line)
            # 未開始且未命中就繼續掃下一行
            continue
        else:
            # 只在「行首」遇到欄位名才停止
            if STOP_RE.search(line):
                break
            collected.append(line)

    title = " ".join(collected).strip()
    # 再次壓縮空白，防止跨行拼接留下多餘空格
    title = re.sub(r"\s+", " ", title)
    return title

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
