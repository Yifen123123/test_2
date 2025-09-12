import json
import re
from pathlib import Path

def extract_title(text: str) -> str:
    # 嘗試找「主旨/案由/標題」後的內容
    match = re.search(r"(主旨|案由|標題)\s*[:：]\s*(.+)", text)
    if match:
        return match.group(2).strip()
    # fallback: 用第一行
    return text.strip().splitlines()[0] if text.strip() else ""

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
