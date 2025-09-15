# finetune_e5.py
import os
# 1) 在匯入 transformers / sentence_transformers 之前，強制關掉 TF 分支
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 靜音 TF 若環境還是有它
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import math
from pathlib import Path
from typing import List, Dict

import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, losses, InputExample
from tqdm import tqdm

FINETUNE_FILE = "finetune_pairs.jsonl"     # 由你的資料建置腳本產生
BASE_MODEL    = "intfloat/multilingual-e5-large"
OUTPUT_MODEL  = "./e5_finetuned_triplet"   # 輸出要是「資料夾」
BATCH_SIZE    = 16
EPOCHS        = 3
MARGIN        = 0.25  # Triplet margin，可調 0.2~0.5

def read_pairs(path: str) -> List[Dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items

def build_triplets(file: str) -> List[InputExample]:
    """
    期待每行：
    {"anchor": "...", "positive": ["..."], "hard_negative": ["...", ...]}
    產生 (anchor, positive, negative) 三元組
    """
    raw = read_pairs(file)
    triplets: List[InputExample] = []
    pos_cnt = neg_cnt = 0

    for rec in raw:
        a = rec.get("anchor")
        pos_list = rec.get("positive") or []
        neg_list = rec.get("hard_negative") or []
        # 保證 a/pos/neg 皆存在才組 triplet
        for p in pos_list:
            pos_cnt += 1
            for n in neg_list:
                neg_cnt += 1
                triplets.append(InputExample(texts=[a, p, n]))

    print(f"📦 讀到 {len(raw)} 行；三元組數={len(triplets)}（pos對={pos_cnt}, neg候選={neg_cnt}）")
    return triplets

def main():
    # 0) 基本健檢
    finetune_path = Path(FINETUNE_FILE)
    if not finetune_path.exists():
        raise FileNotFoundError(f"{FINETUNE_FILE} 不存在，請先生成微調資料。")

    out_dir = Path(OUTPUT_MODEL)
    # 明確保證是資料夾且可寫入
    out_dir.mkdir(parents=True, exist_ok=True)
    if out_dir.exists() and out_dir.is_file():
        raise NotADirectoryError(f"OUTPUT_MODEL 指向檔案而非資料夾：{out_dir}")

    # 1) 判斷設備（CPU 預設）
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 使用設備: {device}")
    print(f"📝 將輸出模型到資料夾: {out_dir.resolve()}")

    # 2) 載入 base encoder（強制 PyTorch）
    model = SentenceTransformer(BASE_MODEL, device=device)

    # 3) 準備訓練三元組
    triplets = build_triplets(FINETUNE_FILE)
    if len(triplets) < 2:
        raise ValueError("Triplet 樣本太少（<2）。請檢查 finetune_pairs.jsonl 是否同時含 positive 與 hard_negative。")

    # 4) DataLoader（CPU 友善設定）
    # drop_last=True 避免最後批只有 1 筆導致 loss 無法計算
    eff_batch = min(BATCH_SIZE, max(2, len(triplets)))  # 保證 batch >= 2 且不超過資料數
    dataloader = DataLoader(
        triplets,
        shuffle=True,
        batch_size=eff_batch,
        pin_memory=False,     # CPU-only 明確關閉
        drop_last=True
    )
    steps_per_epoch = len(dataloader)
    if steps_per_epoch == 0:
        raise ValueError(
            f"dataloader 產生 0 個批次。triplets={len(triplets)}, batch_size={eff_batch}\n"
            f"→ 請降低 BATCH_SIZE 或增加資料量。"
        )
    print(f"🧮 steps_per_epoch={steps_per_epoch} (batch_size={eff_batch})")

    # 5) Triplet Loss（以 cosine distance）
    train_loss = losses.TripletLoss(
        model=model,
        distance_metric=losses.SiameseDistanceMetric.COSINE_DISTANCE,
        triplet_margin=MARGIN
    )

    # 6) 訓練（關閉 AMP；只用 PyTorch）
    total_steps = EPOCHS * steps_per_epoch
    warmup_steps = max(0, int(0.1 * total_steps))
    print(f"🚀 開始訓練：epochs={EPOCHS}, total_steps={total_steps}, warmup_steps={warmup_steps}")

    model.fit(
        train_objectives=[(dataloader, train_loss)],
        epochs=EPOCHS,
        warmup_steps=warmup_steps,
        output_path=str(out_dir),
        show_progress_bar=True,
        use_amp=False  # 關閉自動混合精度，避免跨後端問題
    )

    print(f"✅ 微調完成，模型已輸出到: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
