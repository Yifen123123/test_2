# build_index.py
import json
from pathlib import Path
from typing import Iterator, List, Dict

import chromadb
from sentence_transformers import SentenceTransformer

# ====== 基本設定 ======
DATA_JSONL = "train.jsonl"              # 你剛輸出的訓練 jsonl
DB_PATH    = "chroma_db"                # 向量庫持久化目錄
COLLECTION = "gov_letters"             # 集合名稱（同一庫只要一致）

# HNSW 參數（可依量級微調）
HNSW_SPACE = "cosine"                  # 餘弦相似度（搭配 L2 normalize）
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200

# 批次設定
BATCH_SIZE = 256

# ====== 載入 embedding 模型（CPU 也可）
encoder = SentenceTransformer("intfloat/multilingual-e5-large")

def load_jsonl(path: str) -> Iterator[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def batched(iterable: Iterator[Dict], n: int) -> Iterator[List[Dict]]:
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch

def main():
    # 1) 啟用持久化客戶端
    client = chromadb.PersistentClient(path=DB_PATH)

    # 2) 建立/取得集合（指定 HNSW + cosine；並可設定 M/efConstruction）
    collection = client.get_or_create_collection(
        name=COLLECTION,
        metadata={
            "hnsw:space": HNSW_SPACE,
            "hnsw:M": HNSW_M,
            "hnsw:efConstruction": HNSW_EF_CONSTRUCTION,
        },
    )

    # 3) 批次寫入
    total = 0
    for batch in batched(load_jsonl(DATA_JSONL), BATCH_SIZE):
        ids = [rec["id"] for rec in batch]  # 確保唯一（重複會報錯；要覆蓋可改用 upsert）
        docs = [rec["title"] for rec in batch]
        metas = [{"label": rec["label"]} for rec in batch]

        # 3a) 生成向量（**務必 L2 normalize**）
        embs = encoder.encode(docs, normalize_embeddings=True, batch_size=64).tolist()

        # 3b) 寫入 Chroma
        collection.add(ids=ids, documents=docs, embeddings=embs, metadatas=metas)
        total += len(batch)
        print(f"Indexed so far: {total}")

    # 4) 驗證：集合大小 & 簡單查詢
    count = collection.count()
    print(f"Collection '{COLLECTION}' ready. Total vectors: {count}")

    # 測試一筆查詢（可改成你的文字）
    demo_query = "請於文到期限內查復保單相關資料"
    q_emb = encoder.encode([demo_query], normalize_embeddings=True).tolist()
    res = collection.query(query_embeddings=q_emb, n_results=5)
    print("Top-5 labels:", [m["label"] for m in res["metadatas"][0]])

if __name__ == "__main__":
    main()
