import json
from pathlib import Path
import joblib
import numpy as np
from collections import defaultdict, Counter
from sklearn.metrics import accuracy_score, classification_report
from sentence_transformers import SentenceTransformer
import pandas as pd

TEST_JSONL = "test.jsonl"
MODEL_FILE = "svm_clf.joblib"

def read_jsonl(p):
    for line in Path(p).read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)

def main():
    # 1) 載入模型與 E5
    bundle = joblib.load(MODEL_FILE)
    clf = bundle["clf"]
    e5_name = bundle["e5_name"]
    encoder = SentenceTransformer(e5_name)

    # 2) 讀 test.jsonl
    titles, y_true = [], []
    for rec in read_jsonl(TEST_JSONL):
        titles.append(rec["title"])
        y_true.append(rec["label"])

    # 3) 文字 → 向量（同一 E5；可再次 normalize_embeddings=True）
    X_emb = encoder.encode(titles, normalize_embeddings=True, batch_size=64)

    # 4) 預測（含機率）
    proba = clf.predict_proba(X_emb)      # [n_samples, n_classes]
    classes = clf.classes_
    y_pred = classes[proba.argmax(1)]

    # 5) 指標
    acc = accuracy_score(y_true, y_pred)
    print("Accuracy:", round(acc, 4))
    print("\nClassification report (macro avg 重點看):")
    print(classification_report(y_true, y_pred, digits=4))

    # 6) 簡單誤判流向表（Top-3）
    conf = defaultdict(Counter)
    for t, p in zip(y_true, y_pred):
        conf[t][p] += 1
    rows = []
    for c in sorted(conf.keys()):
        wrong = [(lb, cnt) for lb, cnt in conf[c].items() if lb != c]
        wrong.sort(key=lambda x: x[1], reverse=True)
        row = {"True Label": c}
        for i, (lb, cnt) in enumerate(wrong[:3]):
            row[f"Top{i+1}"] = f"{lb} ({cnt})"
        rows.append(row)
    df = pd.DataFrame(rows).fillna("-")
    print("\n錯誤去向（Top-3）：")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
