# build_finetune_data_hybrid.py
import json
import random
from collections import defaultdict, Counter
from pathlib import Path

# === 輸入/輸出 ===
TRAIN_JSONL = "train.jsonl"
PRED_JSONL  = "predictions.jsonl"  # 來源於你先前的評估輸出：每行含 id, title, true, pred
OUT_JSONL   = "finetune_pairs.jsonl"

# === 是否使用 Chroma 近鄰（僅查 train 向量庫）===
USE_CHROMA = True
CHROMA_DB_PATH = "chroma_db"
CHROMA_COLLECTION = "gov_letters"
E5_MODEL_NAME = "intfloat/multilingual-e5-large"

# === 手動混淆對（可調整/擴充）===
CONFUSABLE = {
    "保單查詢": ["保單註記", "保單查詢＋註記"],
    "保單註記": ["保單查詢", "保單查詢＋註記", "通知函"],
    "保單查詢＋註記": ["保單查詢", "保單註記"],
    "收取＋撤銷": ["收取令"],
    "收取令": ["收取＋撤銷"],
}

# === 每個 anchor 的抽樣量（可調）===
POS_PER_ANCHOR = 2
HN_FROM_CONFUSABLE_PER_ANCHOR = 2
HN_FROM_CHROMA_PER_ANCHOR = 3
HN_FROM_PRED_LABELS_PER_ANCHOR = 2   # 依 predictions 找到的「常見誤判類別」來補負例

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

def read_jsonl(p):
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def load_by_label(jsonl_path):
    by_label = defaultdict(list)
    for rec in read_jsonl(jsonl_path):
        by_label[rec["label"]].append(rec["title"])
    return by_label

def load_titles_with_labels(jsonl_path):
    items = []
    for rec in read_jsonl(jsonl_path):
        items.append((str(rec.get("id")), rec["title"], rec["label"]))
    return items

def load_pred_label_map(pred_path):
    """
    讀 predictions.jsonl（不讀 test.jsonl 正文），
    回傳：mislabel_stats[true_label] = Counter({pred_label: count, ...})
    僅用「誤判的類別名稱」訊號來決定訓練時該強化哪些負例類別；負例文本仍從 train 取。
    """
    stats = defaultdict(Counter)
    if not Path(pred_path).exists():
        return stats
    for rec in read_jsonl(pred_path):
        true_lb = rec.get("true")
        pred_lb = rec.get("pred")
        if true_lb and pred_lb and pred_lb != true_lb:
            stats[true_lb][pred_lb] += 1
    return stats

def safe_sample(lst, k, exclude=None):
    if not lst:
        return []
    pool = lst if exclude is None else [x for x in lst if x != exclude]
    if not pool:
        return []
    if k >= len(pool):
        return list(pool)
    return random.sample(pool, k)

# --- Chroma 近鄰：僅查 train 向量庫 ---
def setup_chroma():
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        col = client.get_collection(CHROMA_COLLECTION)
        encoder = SentenceTransformer(E5_MODEL_NAME)
        return col, encoder
    except Exception as e:
        print(f"[Chroma] 初始化失敗：{e}")
        return None, None

def chroma_hard_negs(col, encoder, text, true_label, topn=20):
    try:
        q = encoder.encode([text], normalize_embeddings=True).tolist()
        res = col.query(query_embeddings=q, n_results=topn)
        metas = res.get("metadatas", [[]])[0]
        docs  = res.get("documents", [[]])[0]
        cand = []
        for m, d in zip(metas, docs):
            lb = m.get("label")
            if lb and lb != true_label:
                cand.append(d)
        return cand
    except Exception:
        return []

def main():
    if not Path(TRAIN_JSONL).exists():
        raise FileNotFoundError(f"{TRAIN_JSONL} 不存在")
    if not Path(PRED_JSONL).exists():
        print(f"⚠️ 找不到 {PRED_JSONL}，將不使用預測誤判訊號。")

    # 1) 讀取 train
    train_by_label = load_by_label(TRAIN_JSONL)
    train_items = load_titles_with_labels(TRAIN_JSONL)  # [(id, title, label)]

    # 2) 讀取 predictions.jsonl → 形成 true_label → 常見誤判的 pred_labels
    mislabel_stats = load_pred_label_map(PRED_JSONL)
    # 轉成：true_label -> list of pred_labels（依出現次數排序）
    mislabel_labels = {lb: [plb for plb, _cnt in cnts.most_common()] 
                       for lb, cnts in mislabel_stats.items()}

    # 3) Chroma（僅查 train 向量庫）
    col = enc = None
    if USE_CHROMA:
        col, enc = setup_chroma()
        if col is None:
            print("⚠️ 關閉 USE_CHROMA（初始化失敗）")
            USE_CHROMA = False

    # 4) 為每個 train 的句子產生 (anchor, positives, hard_negatives)
    out_cnt = 0
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for _id, anchor_text, true_lb in train_items:
            # positives：同類（train）
            positives = safe_sample(train_by_label.get(true_lb, []), POS_PER_ANCHOR, exclude=anchor_text)

            # hard negatives 來源集合
            neg_set = set()

            # a) 手動混淆對
            for clb in CONFUSABLE.get(true_lb, []):
                for t in safe_sample(train_by_label.get(clb, []), HN_FROM_CONFUSABLE_PER_ANCHOR, exclude=anchor_text):
                    neg_set.add(t)

            # b) 由 predictions.jsonl 觀察到的「常見誤判類別」
            #    只拿「類別名稱」當訊號，實際負例文本仍從 train 取
            for plb in mislabel_labels.get(true_lb, [])[:3]:  # 最常見的前三個誤判類別
                for t in safe_sample(train_by_label.get(plb, []), HN_FROM_PRED_LABELS_PER_ANCHOR, exclude=anchor_text):
                    neg_set.add(t)

            # c) Chroma 近鄰（異類）
            if USE_CHROMA and col and enc:
                cands = chroma_hard_negs(col, enc, anchor_text, true_lb, topn=20)
                for t in cands[:HN_FROM_CHROMA_PER_ANCHOR]:
                    neg_set.add(t)

            negatives = list(neg_set)

            if positives and negatives:
                rec = {"anchor": anchor_text, "positive": positives, "hard_negative": negatives}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_cnt += 1

    # 5) 摘要
    print(f"✅ 已輸出 {out_cnt} 行到 {OUT_JSONL}")
    if out_cnt == 0:
        print("⚠️ 沒有任何 anchor 同時具備 positive 與 hard_negative。")
        print("   建議：擴充 CONFUSABLE、放寬每 anchor 抽樣數，或啟用/修正 Chroma。")

    # 額外列出每個 true_label 的「來自 predictions 的常見誤判方向」
    if mislabel_stats:
        print("\n🔎 由 predictions.jsonl 推得的常見誤判方向（true -> pred: count）：")
        for lb in sorted(mislabel_stats.keys()):
            row = ", ".join(f"{plb}:{cnt}" for plb, cnt in mislabel_stats[lb].most_common())
            print(f"  {lb} -> {row}")

if __name__ == "__main__":
    main()
