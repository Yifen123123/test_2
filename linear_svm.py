import json
from pathlib import Path
import joblib
from sklearn.svm import LinearSVC
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import make_scorer, f1_score
from sentence_transformers import SentenceTransformer

TRAIN_JSONL = "train.jsonl"
MODEL_FILE  = "svm_clf.joblib"
E5_NAME     = "intfloat/multilingual-e5-large"

def read_jsonl(p):
    for line in Path(p).read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)

def load_xy(jsonl_path):
    X_text, y = [], []
    for rec in read_jsonl(jsonl_path):
        X_text.append(rec["title"])
        y.append(rec["label"])
    return X_text, y

def main():
    # 1) 載入資料
    X_text, y = load_xy(TRAIN_JSONL)

    # 2) 文字 → 向量（E5，L2 正規化交給管線也可，這裡先做一致化）
    encoder = SentenceTransformer(E5_NAME)
    X_emb = encoder.encode(X_text, normalize_embeddings=True, batch_size=64)

    # 3) 建管線：L2 normalize → LinearSVC，外掛機率校準
    base = make_pipeline(
        Normalizer(norm="l2"),
        LinearSVC(class_weight="balanced", max_iter=5000)
    )

    # 4) 在訓練集內做 5-fold grid search（只在訓練資料上調 C）
    params = {"linearsvc__C": [0.1, 0.5, 1.0, 2.0, 5.0]}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scorer = make_scorer(f1_score, average="macro")
    gs = GridSearchCV(base, params, scoring=scorer, cv=cv, n_jobs=-1, refit=True)
    gs.fit(X_emb, y)
    print("Best params:", gs.best_params_, "CV macro-F1:", round(gs.best_score_, 4))

    # 5) 用最佳管線做機率校準（再做一次 5-fold 內部校準）
    clf = CalibratedClassifierCV(gs.best_estimator_, cv=5, method="sigmoid")
    clf.fit(X_emb, y)

    # 6) 存模型（裡面含 Normalizer 與 LinearSVC）
    joblib.dump({"clf": clf, "e5_name": E5_NAME}, MODEL_FILE)
    print(f"Saved model to {MODEL_FILE}")

if __name__ == "__main__":
    main()
