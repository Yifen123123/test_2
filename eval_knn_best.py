# eval_knn_best.py
import json
import time
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import chromadb
from sklearn.metrics import f1_score, accuracy_score
from sentence_transformers import SentenceTransformer

# ===== 基本設定 =====
DB_PATH     = "chroma_db"
COLLECTION  = "gov_letters"
TEST_JSONL  = "test.jsonl"
MODEL_NAME  = "intfloat/multilingual-e5-large"

# 固定最佳參數（你剛找到的組合）
BEST_PARAMS = {"K": 3, "P": 1.0, "TAU": None, "MARGIN": 0.0}

SAVE_PREDICTIONS = True
PRED_JSONL = "predictions.jsonl"

# ===== 輔助 =====
def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def load_test(path):
    titles, labels, ids = [], [], []
    for rec in read_jsonl(path):
        titles.append(rec["title"])
        labels.append(rec["label"])
        ids.append(rec.get("id", None))
    return titles, labels, ids

# ===== 初始化模型與 DB 連線 =====
encoder = SentenceTransformer(MODEL_NAME)
client  = chromadb.PersistentClient(path=DB_PATH)
col     = client.get_collection(COLLECTION)

def predict_one(title, K=5, P=2.0, TAU=None, MARGIN=0.0):
    """
    回傳: pred_label, sims(list), knn_labels(list), t_enc, t_qry
    """
    t0 = time.perf_counter()
    q = encoder.encode([title], normalize_embeddings=True).tolist()
    t1 = time.perf_counter()
    res = col.query(query_embeddings=q, n_results=K)
    t2 = time.perf_counter()

    labels = [m.get("label") for m in res["metadatas"][0]]

    if "distances" in res and res["distances"]:
        dists = res["distances"][0]
        sims  = [1.0 - d for d in dists]   # cosine distance -> similarity
    else:
        sims = [1.0] * len(labels)

    # 拒答規則
    s_sorted = sorted(sims, reverse=True) + [0.0, 0.0]
    if TAU is not None and s_sorted[0] < TAU:
        return "unknown", sims, labels, (t1 - t0), (t2 - t1)
    if MARGIN > 0.0 and (s_sorted[0] - s_sorted[1]) < MARGIN:
        return "unknown", sims, labels, (t1 - t0), (t2 - t1)

    # 加權投票
    score = defaultdict(float)
    for lb, s in zip(labels, sims):
        score[lb] += (s ** P)
    pred = max(score.items(), key=lambda x: x[1])[0]
    return pred, sims, labels, (t1 - t0), (t2 - t1)

def evaluate_titles(titles, y_true, ids=None, **params):
    y_pred = []
    conf = defaultdict(Counter)
    enc_times, qry_times = [], []
    pred_rows = []

    for idx, (title, tlabel) in enumerate(zip(titles, y_true)):
        pred, sims, knn_labels, t_enc, t_qry = predict_one(title, **params)
        y_pred.append(pred)
        conf[tlabel][pred] += 1
        enc_times.append(t_enc)
        qry_times.append(t_qry)

        if ids:
            pred_rows.append({"id": ids[idx], "title": title, "true": tlabel, "pred": pred})
        else:
            pred_rows.append({"id": idx, "title": title, "true": tlabel, "pred": pred})

    # 覆蓋率與 macro-F1（忽略 unknown）
    mask = np.array([p != "unknown" for p in y_pred])
    coverage = float(mask.mean()) if len(mask) else 0.0
    if mask.any():
        macro_f1 = f1_score(np.array(y_true)[mask], np.array(y_pred)[mask], average="macro")
    else:
        macro_f1 = 0.0

    # 整體 accuracy（含 unknown 當錯）
    acc = accuracy_score(y_true, y_pred)

    # 各類正確率
    per_class_acc = {}
    for c in sorted(set(y_true)):
        n_true = sum(1 for t in y_true if t == c)
        per_class_acc[c] = conf[c][c] / n_true if n_true else 0.0

    # 錯誤去向 Top-3 表
    rows = []
    for c in sorted(conf.keys()):
        wrong = [(lb, cnt) for lb, cnt in conf[c].items() if lb != c]
        wrong.sort(key=lambda x: x[1], reverse=True)
        row = {"True Label": c}
        for i, (lb, cnt) in enumerate(wrong[:3]):
            row[f"Top{i+1}"] = f"{lb} ({cnt})"
        rows.append(row)
    wrong_df = pd.DataFrame(rows).fillna("-")

    # 時間統計
    def stats(a):
        a_sorted = sorted(a)
        n = len(a_sorted)
        p50 = a_sorted[int(0.50*(n-1))] if n else 0.0
        p95 = a_sorted[int(0.95*(n-1))] if n else 0.0
        return (sum(a_sorted)/n if n else 0.0, p50, p95)

    enc_avg, enc_p50, enc_p95 = stats(enc_times)
    qry_avg, qry_p50, qry_p95 = stats(qry_times)

    return {
        "y_pred": y_pred,
        "pred_rows": pred_rows,
        "conf": conf,
        "macro_f1": macro_f1,
        "accuracy": acc,
        "coverage": coverage,
        "per_class_acc": per_class_acc,
        "wrong_df": wrong_df,
        "enc_stats": (enc_avg, enc_p50, enc_p95),
        "qry_stats": (qry_avg, qry_p50, qry_p95),
    }

def main():
    titles, y_true, ids = load_test(TEST_JSONL)

    t0 = time.perf_counter()
    res = evaluate_titles(titles, y_true, ids=ids, **BEST_PARAMS)
    t1 = time.perf_counter()

    print("\n=== 最終評估（固定最佳參數） ===")
    print(f"Best params: {BEST_PARAMS}")
    print(f"Accuracy: {res['accuracy']:.4f} | Macro-F1: {res['macro_f1']:.4f} | Coverage: {res['coverage']:.4f}\n")

    print("各類正確率：")
    for c in sorted(res["per_class_acc"], key=lambda x: res["per_class_acc"][x], reverse=True):
        print(f"  {c}: {res['per_class_acc'][c]:.4f}")

    print("\n錯誤去向（Top-3 誤判方向表格）：")
    print(res["wrong_df"].to_string(index=False))

    enc_avg, enc_p50, enc_p95 = res["enc_stats"]
    qry_avg, qry_p50, qry_p95 = res["qry_stats"]
    print("\n=== 執行時間（秒）===")
    print(f"總耗時: {t1 - t0:.3f}")
    print(f"編碼  平均: {enc_avg:.4f} | P50: {enc_p50:.4f} | P95: {enc_p95:.4f}")
    print(f"查詢  平均: {qry_avg:.4f} | P50: {qry_p50:.4f} | P95: {qry_p95:.4f}")

    if SAVE_PREDICTIONS:
        with open(PRED_JSONL, "w", encoding="utf-8") as f:
            for row in res["pred_rows"]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\n已輸出逐筆預測到: {PRED_JSONL}")

if __name__ == "__main__":
    main()
