# finetune_e5.py
import json
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, losses, InputExample

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
            # 這裡也用 InputExample，batch 內會自動當 negative
            examples.append(InputExample(texts=[anchor, neg]))
    return examples

def main():
    model = SentenceTransformer(BASE_MODEL)
    examples = build_examples(FINETUNE_FILE)
    dataloader = DataLoader(examples, shuffle=True, batch_size=16)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    model.fit(
        train_objectives=[(dataloader, train_loss)],
        epochs=3,
        warmup_steps=10,
        output_path=OUTPUT_MODEL
    )
    print(f"✅ 微調完成，已輸出到 {OUTPUT_MODEL}")

if __name__ == "__main__":
    main()
