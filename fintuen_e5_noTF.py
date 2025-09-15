# finetune_e5_mnrl.py
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"   # 避免 transformers 走 TF 分支
os.environ["USE_TF"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_MODE"] = "disabled"    # 關掉 wandb 互動

import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, losses, InputExample

# 建議改成純英文「絕對」路徑
ROOT          = r"C:\mlproj"
FINETUNE_FILE = str(Path(ROOT) / "finetune_pairs.jsonl")
OUTPUT_DIR    = str(Path(ROOT) / "e5_finetuned_mnrl")
BASE_MODEL    = "intfloat/multilingual-e5-large"

BATCH_SIZE = 16
EPOCHS     = 3
WARMUP_PCT = 0.1

def read_pairs(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def build_positive_examples(file):
    """只建立正例對：texts=[anchor, positive]；hard_negative 不要放進來！"""
    examples = []
    pos_pairs = 0
    for rec in read_pairs(file):
        a = rec.get("anchor")
        for p in rec.get("positive", []) or []:
            if a and p:
                examples.append(InputExample(texts=[a, p]))
                pos_pairs += 1
    return examples, pos_pairs

def main():
    # 目錄防呆
    Path(ROOT).mkdir(parents=True, exist_ok=True)
    if not Path(FINETUNE_FILE).exists():
        raise FileNotFoundError(f"找不到 {FINETUNE_FILE}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 device = {device}")

    # 1) 準備資料（只正例）
    examples, pos_pairs = build_positive_examples(FINETUNE_FILE)
    if len(examples) < 2:
        raise ValueError("有效的正例對不足（<2）。請檢查 finetune_pairs.jsonl 的 positive 欄位。")
    print(f"📂 正例對數量 = {pos_pairs}（實際 examples = {len(examples)}）")

    # 2) DataLoader（CPU 友善）
    eff_bs = min(BATCH_SIZE, max(2, len(examples)))
    dataloader = DataLoader(
        examples,
        shuffle=True,
        batch_size=eff_bs,
        pin_memory=False,   # CPU：關閉
        drop_last=False     # 小資料別丟最後一批
    )
    steps_per_epoch = len(dataloader)
    print(f"🧮 steps_per_epoch = {steps_per_epoch} | batch_size = {eff_bs}")
    if steps_per_epoch == 0:
        raise ValueError("steps_per_epoch=0，請降低 BATCH_SIZE 或增加樣本。")

    # 3) 模型 & Loss
    model = SentenceTransformer(BASE_MODEL, device=device)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    total_steps  = EPOCHS * steps_per_epoch
    warmup_steps = max(0, int(WARMUP_PCT * total_steps))
    print(f"🚀 訓練開始：epochs={EPOCHS}, total_steps={total_steps}, warmup_steps={warmup_steps}")

    # 4) 訓練（不落地，結束後再手動保存）
    model.fit(
        train_objectives=[(dataloader, train_loss)],
        epochs=EPOCHS,
        warmup_steps=warmup_steps,
        output_path=None,        # 不存中途檔，避免 TF/路徑干擾
        checkpoint_path=None,
        show_progress_bar=True,
        use_amp=False
    )

    # 5) 保存最終模型（到純英文路徑）
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    model.save(OUTPUT_DIR)
    print(f"✅ 微調完成，模型已保存：{OUTPUT_DIR}")

if __name__ == "__main__":
    main()
