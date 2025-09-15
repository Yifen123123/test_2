# finetune_e5.py
import json
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, losses, InputExample
from tqdm import tqdm

FINETUNE_FILE = "finetune_pairs.jsonl"
BASE_MODEL    = "intfloat/multilingual-e5-large"
OUTPUT_MODEL  = "./e5_finetuned"

def read_pairs(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def build_examples(file):
    examples = []
    for rec in read_pairs(file):
        anchor = rec["anchor"]
        for pos in rec.get("positive", []):
            examples.append(InputExample(texts=[anchor, pos]))
        for neg in rec.get("hard_negative", []):
            # 這裡仍用 InputExample，batch 內會自動當 negative
            examples.append(InputExample(texts=[anchor, neg]))
    return examples

def main():
    # 1) 判斷設備
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 使用設備: {device}")

    # 2) 載入 base model
    model = SentenceTransformer(BASE_MODEL, device=device)

    # 3) 準備訓練資料
    examples = build_examples(FINETUNE_FILE)
    print(f"📂 總共載入 {len(examples)} 筆訓練對")

    # 4) DataLoader，pin_memory 自動依環境設定
    dataloader = DataLoader(
        examples,
        shuffle=True,
        batch_size=16,
        pin_memory=torch.cuda.is_available()
    )

    # 5) Loss
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # 6) 訓練
    model.fit(
        train_objectives=[(dataloader, train_loss)],
        epochs=3,
        warmup_steps=10,
        output_path=OUTPUT_MODEL,
        show_progress_bar=True  # 內建進度條
    )

    print(f"✅ 微調完成，模型已輸出到 {OUTPUT_MODEL}")

if __name__ == "__main__":
    main()
