# build_finetune_data.py
import json
import random
from collections import defaultdict
from pathlib import Path

TRAIN_JSONL = "train.jsonl"
TEST_JSONL  = "test.jsonl"
PRED_JSONL  = "predictions.jsonl"
OUT_JSONL   = "finetune_pairs.jsonl"

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def build_positive_pairs(train_file):
    """同類別產生正例 pairs"""
    by_label = defaultdict(list)
    for rec in read_jsonl(train_file):
        by_label[rec["label"]].append(rec["title"])

    pairs = []
    for lb, titles in by_label.items():
        if len(titles) < 2: 
            continue
        # 隨機抽一些正例 pairs
        for _ in range(min(20, len(titles))):
            a, b = random.sample(titles, 2)
            pairs.append({"anchor": a, "positive": [b], "hard_negative": []})
    return pairs

def build_hard_negatives(test_file, pred_file):
    """從誤判結果生成 hard negatives"""
    preds = {rec["id"]: rec for rec in read_jsonl(pred_file)}
    pairs = []
    for rec in read_jsonl(test_file):
        rid, title, true = rec["id"], rec["title"], rec["label"]
        pred = preds.get(rid, {}).get("pred")
        if pred and pred != true:
            pairs.append({"anchor": title, "positive": [], "hard_negative": [pred]})
    return pairs

def main():
    pos_pairs  = build_positive_pairs(TRAIN_JSONL)
    hard_pairs = build_hard_negatives(TEST_JSONL, PRED_JSONL)
    all_pairs  = pos_pairs + hard_pairs

    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for rec in all_pairs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"✅ 已輸出 {len(all_pairs)} 筆訓練對到 {OUT_JSONL}")

if __name__ == "__main__":
    main()
