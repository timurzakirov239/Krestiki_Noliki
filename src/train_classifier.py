from __future__ import annotations

from pathlib import Path
import json
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

from common import load_vivino, make_features


BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"


def price_to_class(price: float, q33: float, q66: float) -> str:
    if price <= q33:
        return "дёшево"
    if price <= q66:
        return "средне"
    return "дорого"


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_vivino()

    X, numeric_cols, cat_cols, text_col = make_features(df, use_name_text=False)

    # Split by indices to compute quantiles ONLY on train
    idx = np.arange(len(df))
    idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=42)

    df_train = df.iloc[idx_train].copy()
    df_test = df.iloc[idx_test].copy()

    q33 = float(df_train["Price"].quantile(0.33))
    q66 = float(df_train["Price"].quantile(0.66))

    y_train = df_train["Price"].map(lambda p: price_to_class(float(p), q33, q66)).astype(str).values
    y_test = df_test["Price"].map(lambda p: price_to_class(float(p), q33, q66)).astype(str).values

    X_train = X.iloc[idx_train].copy()
    X_test = X.iloc[idx_test].copy()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), numeric_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore")),
            ]), cat_cols),
        ],
        remainder="drop"
    )

    model = Pipeline(steps=[
        ("prep", preprocessor),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    print("\n=== КЛАССИФИКАЦИЯ: ценовой сегмент (дёшево/средне/дорого) ===")
    print(f"Пороги цены (посчитаны по train): q33={q33:.2f} €, q66={q66:.2f} €\n")
    print(classification_report(y_test, pred, zero_division=0))

    labels_order = ["дёшево", "средне", "дорого"]
    print("Матрица ошибок:\n", confusion_matrix(y_test, pred, labels=labels_order))

    out_model = MODELS_DIR / "vivino_classifier_priceclass.joblib"
    joblib.dump(model, out_model)

    thresholds = {"q33": q33, "q66": q66, "classes": labels_order}
    out_thr = MODELS_DIR / "vivino_priceclass_thresholds.json"
    out_thr.write_text(json.dumps(thresholds, ensure_ascii=False, indent=2), encoding="utf-8")

    meta = {
        "task": "price_classification",
        "target": "PriceClass (derived from Price quantiles on train)",
        "thresholds": thresholds,
        "numeric_features": numeric_cols,
        "categorical_features": cat_cols,
        "rows": int(len(df)),
    }
    out_meta = MODELS_DIR / "vivino_classifier_priceclass_meta.json"
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Модель сохранена:", out_model)
    print("Пороги сохранены:", out_thr)
    print("Метаданные сохранены:", out_meta)


if __name__ == "__main__":
    main()
