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
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from common import load_vivino, make_features


BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_vivino()

    # Features (without text Name for simplicity)
    X, numeric_cols, cat_cols, text_col = make_features(df, use_name_text=False)

    # Target: Price (log-transform because of strong outliers)
    y_log = np.log1p(df["Price"].astype(float).values)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )

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
        ("reg", Ridge(alpha=1.0))
    ])

    model.fit(X_train, y_train)
    pred_log = model.predict(X_test)

    pred_price = np.expm1(pred_log)
    true_price = np.expm1(y_test)

    mae = mean_absolute_error(true_price, pred_price)
    rmse = float(np.sqrt(mean_squared_error(true_price, pred_price)))
    r2 = r2_score(true_price, pred_price)

    print("\n=== РЕГРЕССИЯ: прогноз цены (Price) ===")
    print(f"MAE:  {mae:.2f} €")
    print(f"RMSE: {rmse:.2f} €")
    print(f"R²:   {r2:.3f}")

    out_model = MODELS_DIR / "vivino_regressor_price.joblib"
    joblib.dump(model, out_model)

    meta = {
        "task": "price_regression",
        "target": "Price",
        "log_target": True,
        "numeric_features": numeric_cols,
        "categorical_features": cat_cols,
        "metrics": {"mae_eur": float(mae), "rmse_eur": float(rmse), "r2": float(r2)},
        "rows": int(len(df)),
    }
    out_meta = MODELS_DIR / "vivino_regressor_price_meta.json"
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Модель сохранена:", out_model)
    print("Метаданные сохранены:", out_meta)


if __name__ == "__main__":
    main()
