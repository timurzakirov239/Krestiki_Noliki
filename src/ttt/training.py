from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

from .generate_dataset import generate


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "ttt"
MODELS_DIR = BASE_DIR / "models"
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class LevelConfig:
    hidden_layers: tuple[int, ...]
    epochs: int
    batch_size: int
    lr: float
    alpha: float


CONFIG = {
    "junior": LevelConfig(hidden_layers=(16,), epochs=25, batch_size=128, lr=0.08, alpha=1e-4),
    "mid":    LevelConfig(hidden_layers=(32, 32), epochs=45, batch_size=128, lr=0.05, alpha=1e-4),
    "senior": LevelConfig(hidden_layers=(64, 64), epochs=80, batch_size=128, lr=0.03, alpha=1e-4),
}


def load_or_generate_dataset(level: str, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    path = DATA_DIR / f"ttt_{level}.npz"
    if path.exists():
        data = np.load(path)
        return data["X"], data["y"]

    X, y = generate(level=level, seed=seed, augment=True)
    np.savez_compressed(path, X=X, y=y)
    return X, y


def train_level(level: str, seed: int = 42, progress_cb=None, save: bool = True):
    cfg = CONFIG[level]
    X, y = load_or_generate_dataset(level, seed=seed)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    clf = MLPClassifier(
        hidden_layer_sizes=cfg.hidden_layers,
        activation="relu",
        solver="sgd",
        learning_rate_init=cfg.lr,
        alpha=cfg.alpha,
        momentum=0.9,
        nesterovs_momentum=True,
        max_iter=1,
        warm_start=False,
        random_state=seed,
    )

    classes = np.arange(9, dtype=np.int64)
    history = {"train_acc": [], "val_acc": []}
    rng = np.random.RandomState(seed)

    for epoch in range(1, cfg.epochs + 1):
        idx = rng.permutation(len(X_train))
        X_sh = X_train[idx]
        y_sh = y_train[idx]

        for start in range(0, len(X_sh), cfg.batch_size):
            end = start + cfg.batch_size
            xb = X_sh[start:end]
            yb = y_sh[start:end]
            if epoch == 1 and start == 0:
                clf.partial_fit(xb, yb, classes=classes)
            else:
                clf.partial_fit(xb, yb)

        tr_acc = accuracy_score(y_train, clf.predict(X_train))
        val_acc = accuracy_score(y_val, clf.predict(X_val))

        history["train_acc"].append(float(tr_acc))
        history["val_acc"].append(float(val_acc))

        if progress_cb is not None:
            progress_cb(epoch, float(tr_acc), float(val_acc))

    if save:
        model_path = MODELS_DIR / f"ttt_{level}.joblib"
        joblib.dump(clf, model_path)

        meta = {
            "level": level,
            "config": cfg.__dict__,
            "rows": int(len(X)),
            "history": history,
        }
        meta_path = MODELS_DIR / f"ttt_{level}_meta.json"
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return clf, history
