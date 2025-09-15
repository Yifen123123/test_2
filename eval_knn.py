import json
import time
from collections import defaultdict, Counter
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

# ===== 基本設定 =====
DB_PATH     = "chroma_db"
COLLECTION  = "gov_letters"
TEST_JSONL  = "test.jsonl"
MODEL_NAME  = "intfloat/multilingual-e5-large"
K           = 5
P           = 2.0
TAU         = None   # 例如 0.65
MARGIN      = 0.0    # 例如 0.07

encoder = SentenceTransformer(MODEL_NAME)
client  = chromadb.PersistentClient(path=DB_PATH)
col     = client.get_collection(COLLECTION)

def read_jsonl(p):
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip(): yield json.loads(line)

def predict_label(title: str):
    t0 = time.perf_counter()
    q = encoder.encode([title], normalize_embeddings=True).tolist()
    t1 = time.perf_counter()
    res = col.query(query_embeddings=q, n_results=K)
    t2 = time.perf_counter()

    metas = res["metadatas"][0]
    labels = [m.get("label") for m in metas]

    if "distances" in res and res["distances"]:
        dists = res["distances"][0]
        sims  = [1.0 - d for d in dists]
    else:
        sims = [1.0] * len(labels)

    sim_sorted = sorted(sims, reverse=True) + [0.0, 0.0]
    if TAU is not None and sim_sorted[0] < TAU:
        return "unknown", sims, labels, (t1 - t0), (t2 - t1)
    if MARGIN > 0 and (sim_sorted[0] - sim_sorted[1]) < MARGIN:
        return "unknown", sims, labels, (t1 - t0), (t2 - t1)

    score = defaultdict(float)
    for lb, s in zip(labels, sims):
        score[lb] += (s ** P)
    pred = max(score.items(), key=lambda x: x[1])[0]
    return pred, sims, labels, (t1 - t0), (t2 - t1)

def main():
    y_true, y_pred = [], []
    conf = defaultdict(Counter)

    # 計時累計
    t_total0 = time.perf_counter()
    enc_times, qry_times = [], []

    for rec in read_jsonl(TEST_JSONL):
        title = rec["title"]
        true  = rec["label"]
        pred, sims, knn_labels, t_enc, t_qry = predict_label(title)
        y_true.append(true); y_pred.append(pred)
        conf[true][pred] += 1
        enc_times.append(t_enc)
        qry_times.append(t_qry)

    t_total1 = time.perf_counter()

    # 整體/各類準確率
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    overall = correct / max(1, len(y_true))

    per_class_acc = {}
    for c in sorted(set(y_true)):
        n_true = sum(1 for t in y_true if t == c)
        per_class_acc[c] = conf[c][c] / n_true if n_true else 0.0

    print(f"=== 評估結果（k={K}, P={P}, tau={TAU}, margin={MARGIN}）===")
    print(f"總樣本數: {len(y_true)}")
    print(f"整體正確率: {overall:.4f}\n")

    print("各類正確率：")
    for c in sorted(per_class_acc, key=lambda x: per_class_acc[x], reverse=True):
        print(f"  {c}: {per_class_acc[c]:.4f}")

    # 錯誤去向（Top-3）表格
    rows = []
    for c in sorted(conf.keys()):
        wrong = [(lb, cnt) for lb, cnt in conf[c].items() if lb != c]
        wrong.sort(key=lambda x: x[1], reverse=True)
        row = {"True Label": c}
        for i, (lb, cnt) in enumerate(wrong[:3]):
            row[f"Top{i+1}"] = f"{lb} ({cnt})"
        rows.append(row)
    df = pd.DataFrame(rows).fillna("-")
    print("\n錯誤去向（Top-3 誤判方向表格）：")
    print(df.to_string(index=False))

    # 執行時間統計
    def stats(a):
        a_sorted = sorted(a)
        n = len(a_sorted)
        p50 = a_sorted[int(0.50*(n-1))] if n else 0.0
        p95 = a_sorted[int(0.95*(n-1))] if n else 0.0
        return (sum(a_sorted)/n if n else 0.0, p50, p95)

    enc_avg, enc_p50, enc_p95 = stats(enc_times)
    qry_avg, qry_p50, qry_p95 = stats(qry_times)
    total_time = t_total1 - t_total0

    print("\n=== 執行時間（秒）===")
    print(f"總耗時: {total_time:.3f}")
    print(f"編碼  平均: {enc_avg:.4f} | P50: {enc_p50:.4f} | P95: {enc_p95:.4f}")
    print(f"查詢  平均: {qry_avg:.4f} | P50: {qry_p50:.4f} | P95: {qry_p95:.4f}")

if __name__ == "__main__":
    main()
