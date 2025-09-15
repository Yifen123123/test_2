# finetune_e5.py  (修正版：fit 不保存、不產生日誌 → 訓練完手動存)
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
from pathlib import Path
from typing import List, Dict

import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, losses, InputExample

FINETUNE_FILE = "finetune_pairs.jsonl"
BASE_MODEL    = "intfloat/multilingual-e5-large"
OUTPUT_MODEL  = "./e5_finetuned_triplet"   # 目標「資料夾」
BATCH_SIZE    = 16
EPOCHS        = 3
MARGIN        = 0.25

def read_pairs(path: str) -> List[Dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items

def build_triplets(file: str):
    raw = read_pairs(file)
    triplets = []
    pos_cnt = neg_cnt = 0
    for rec in raw:
        a = rec.get("anchor")
        pos_list = rec.get("positive") or []
        neg_list = rec.get("hard_negative") or []
        for p in pos_list:
            pos_cnt += 1
            for n in neg_list:
                neg_cnt += 1
                triplets.append(InputExample(texts=[a, p, n]))
    print(f"📦 讀到 {len(raw)} 行；三元組={len(triplets)}（pos對={pos_cnt}, neg候選={neg_cnt}）")
    return triplets

def main():
    finetune_path = Path(FINETUNE_FILE)
    if not finetune_path.exists():
        raise FileNotFoundError(f"{FINETUNE_FILE} 不存在")

    out_dir = Path(OUTPUT_MODEL)
    out_dir.mkdir(parents=True, exist_ok=True)
    if out_dir.is_file():
        raise NotADirectoryError(f"OUTPUT_MODEL 指到檔案而非資料夾：{out_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 使用設備: {device}")

    model = SentenceTransformer(BASE_MODEL, device=device)

    triplets = build_triplets(FINETUNE_FILE)
    if len(triplets) < 2:
        raise ValueError("Triplet 樣本太少（<2）。請確認同時有 positive 與 hard_negative。")

    eff_bs = min(BATCH_SIZE, max(2, len(triplets)))
    dataloader = DataLoader(
        triplets,
        shuffle=True,
        batch_size=eff_bs,
        pin_memory=False,
        drop_last=True
    )
    steps_per_epoch = len(dataloader)
    if steps_per_epoch == 0:
        raise ValueError(f"steps_per_epoch=0。請降低 BATCH_SIZE 或增加資料量。")
    print(f"🧮 steps_per_epoch={steps_per_epoch} (batch_size={eff_bs})")

    train_loss = losses.TripletLoss(
        model=model,
        distance_metric=losses.SiameseDistanceMetric.COSINE_DISTANCE,
        triplet_margin=MARGIN
    )

    total_steps = EPOCHS * steps_per_epoch
    warmup_steps = max(0, int(0.1 * total_steps))
    print(f"🚀 開始訓練：epochs={EPOCHS}, total_steps={total_steps}, warmup_steps={warmup_steps}")

    # 關鍵：不讓 fit 寫任何東西 → output_path=None, writer_log_dir=None, checkpoint_path=None
    model.fit(
        train_objectives=[(dataloader, train_loss)],
        epochs=EPOCHS,
        warmup_steps=warmup_steps,
        output_path=None,           # 不保存
        writer_log_dir=None,        # 不開 tensorboard
        checkpoint_path=None,       # 不存 checkpoint
        show_progress_bar=True,
        use_amp=False
    )

    # 訓練完再手動保存（PyTorch 分支）
    print("💾 訓練完成，準備保存模型…")
    model.save(str(out_dir))
    print(f"✅ 模型已保存到：{out_dir.resolve()}")

if __name__ == "__main__":
    main()
