import json
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

# ===== 基本設定 =====
DATA_JSONL = "train.jsonl"       # 你剛輸出的訓練資料
DB_PATH = "chroma_db"            # 向量庫目錄
COLLECTION = "gov_letters"       # collection 名稱

# ===== 載入 embedding 模型 =====
encoder = SentenceTransformer("intfloat/multilingual-e5-large")

# ===== 初始化 DB =====
client = chromadb.PersistentClient(path=DB_PATH)

# 建立或取得 collection，只指定 cosine
collection = client.get_or_create_collection(
    name=COLLECTION,
    metadata={"hnsw:space": "cosine"}
)

# ===== 載入 jsonl =====
def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

# ===== 建庫 =====
def build_collection():
    total = 0
    batch_size = 64
    records = list(load_jsonl(DATA_JSONL))
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        ids = [r["id"] for r in batch]
        docs = [r["title"] for r in batch]
        metas = [{"label": r["label"]} for r in batch]
        embs = encoder.encode(docs, normalize_embeddings=True, batch_size=32).tolist()
        collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
        total += len(batch)
        print(f"已索引 {total} 筆")
    print(f"✅ 建庫完成，共 {collection.count()} 筆向量")

# ===== 快速檢查 DB 內容 =====
def inspect_collection(limit=3):
    print(f"\nCollection: {COLLECTION}")
    print("總筆數:", collection.count())
    res = collection.get(limit=limit)
    for i in range(len(res["ids"])):
        print(f"[{i+1}] ID={res['ids'][i]} | Label={res['metadatas'][i].get('label')} | Title={res['documents'][i][:40]}...")

if __name__ == "__main__":
    build_collection()
    inspect_collection()
