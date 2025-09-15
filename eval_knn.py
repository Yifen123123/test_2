import json
import pandas as pd
from collections import defaultdict, Counter
import chromadb
from sentence_transformers import SentenceTransformer

# ===== 基本設定 =====
DB_PATH     = "chroma_db"
COLLECTION  = "gov_letters"
TEST_JSONL  = "test.jsonl"
MODEL_NAME  = "intfloat/multilingual-e5-large"
K           = 5            # 近鄰數
P           = 2.0          # 加權投票：權重 = sim^P
TAU         = None         # 相似度門檻 (e.g., 0.65)；None 表示不啟用 unknown
MARGIN      = 0.0          # top1-top2 margin 門檻；0 表示不啟用

# ===== 載入模型 & 連線資料庫 =====
encoder = SentenceTransformer(MODEL_NAME)
client  = chromadb.PersistentClient(path=DB_PATH)
col     = client.get_collection(COLLECTION)

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def predict_label(title: str):
    q = encoder.encode([title], normalize_embeddings=True).tolist()
    res = col.query(query_embeddings=q, n_results=K)
    metas = res["metadatas"][0]
    labels = [m.get("label") for m in metas]

    # 取得相似度（Chroma cosine distance = 1 - cosine_sim）
    if "distances" in res and res["distances"]:
        dists = res["distances"][0]
        sims  = [1.0 - d for d in dists]
    elif "embeddings" in res and res["embeddings"]:  # 退路（少見）
        # 若回傳的是鄰居向量，這裡就略過，按等權投票
        sims = [1.0] * len(labels)
    else:
        sims = [1.0] * len(labels)

    # 拒答（可選）
    sim_sorted = sorted(sims, reverse=True) + [0.0, 0.0]
    if TAU is not None and sim_sorted[0] < TAU:
        return "unknown", sims, labels
    if MARGIN > 0 and (sim_sorted[0] - sim_sorted[1]) < MARGIN:
        return "unknown", sims, labels

    # 加權投票
    score = defaultdict(float)
    for lb, s in zip(labels, sims):
        score[lb] += (s ** P)
    pred = max(score.items(), key=lambda x: x[1])[0]
    return pred, sims, labels

def main():
    y_true, y_pred = [], []
    conf = defaultdict(Counter)  # conf[true][pred] += 1

    for rec in read_jsonl(TEST_JSONL):
        title = rec["title"]
        true  = rec["label"]
        pred, sims, knn_labels = predict_label(title)
        y_true.append(true); y_pred.append(pred)
        conf[true][pred] += 1

    # 整體正確率
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    overall = correct / max(1, len(y_true))

    # 各類正確率
    per_class_acc = {}
    for c in sorted(set(y_true)):
        n_true = sum(1 for t in y_true if t == c)
        n_hit  = conf[c][c]
        per_class_acc[c] = (n_hit / n_true) if n_true else 0.0

    # 輸出結果
    print(f"=== 評估結果（k={K}, P={P}, tau={TAU}, margin={MARGIN}）===")
    print(f"總樣本數: {len(y_true)}")
    print(f"整體正確率: {overall:.4f}\n")

    print("各類正確率：")
    for c in sorted(per_class_acc, key=lambda x: per_class_acc[x], reverse=True):
        print(f"  {c}: {per_class_acc[c]:.4f}")

    print("\n錯誤去向（Top-3 誤判方向表格）：")

    rows = []
    for c in sorted(conf.keys()):
        wrong = [(lb, cnt) for lb, cnt in conf[c].items() if lb != c]
        wrong.sort(key=lambda x: x[1], reverse=True)
        top3 = {f"Top{i+1}": f"{lb} ({cnt})" for i, (lb, cnt) in enumerate(wrong[:3])}
        row = {"True Label": c} | top3
        rows.append(row)

    df = pd.DataFrame(rows).fillna("-")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
