from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score

from common import load_vivino


BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_vivino()

    # Features for clustering
    df_cl = df.copy()
    df_cl["LogPrice"] = np.log1p(df_cl["Price"].astype(float))
    df_cl["LogRatingsCount"] = np.log1p(df_cl["NumberOfRatings"].astype(float))
    df_cl["YearIsNV"] = df_cl["Year_is_nv"].astype(int)

    numeric_cols = ["Rating", "LogPrice", "LogRatingsCount", "Year_num", "YearIsNV"]
    cat_cols = ["WineType", "Country"]  # keep light for stability

    X = df_cl[numeric_cols + cat_cols].copy()

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

    # 1) Transform to feature matrix
    Z = preprocessor.fit_transform(X)
    n_features = Z.shape[1]

    # 2) Choose SVD components dynamically
    # Need at least 2 for 2D plot, but cannot exceed n_features
    n_components = min(50, n_features)
    if n_components < 2:
        raise ValueError(f"Слишком мало признаков для 2D-визуализации: n_features={n_features}")

    print("\n=== КЛАСТЕРИЗАЦИЯ: подготовка признаков ===")
    print(f"Число признаков после One-Hot: {n_features}")
    print(f"Число компонент SVD: {n_components}")

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    Z_svd = svd.fit_transform(Z)

    scaler = StandardScaler()
    Z_svd_scaled = scaler.fit_transform(Z_svd)

    # 3) Pick k by silhouette on a sample
    rng = np.random.RandomState(42)
    sample_size = min(2000, len(Z_svd_scaled))
    sample_idx = rng.choice(len(Z_svd_scaled), size=sample_size, replace=False)
    Z_sample = Z_svd_scaled[sample_idx]

    best_k = None
    best_score = -1.0

    print("\n=== КЛАСТЕРИЗАЦИЯ: подбор числа кластеров (silhouette) ===")
    for k in range(3, 9):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels_sample = km.fit_predict(Z_sample)
        score = silhouette_score(Z_sample, labels_sample)
        print(f"k={k}: silhouette={score:.4f}")
        if score > best_score:
            best_score = score
            best_k = k

    assert best_k is not None
    print(f"\nВыбрано k={best_k} (лучший silhouette={best_score:.4f})")

    # 4) Final clustering pipeline (fit on raw X)
    cluster_model = Pipeline(steps=[
        ("prep", preprocessor),
        ("svd", svd),
        ("scaler", scaler),
        ("kmeans", KMeans(n_clusters=best_k, random_state=42, n_init=10)),
    ])

    cluster_model.fit(X)
    labels = cluster_model.named_steps["kmeans"].labels_

    # 5) 2D coordinates for plotting: first 2 SVD components
    Z2 = Z_svd[:, :2]

    out_df = df.copy()
    out_df["cluster"] = labels
    out_df["x2d"] = Z2[:, 0]
    out_df["y2d"] = Z2[:, 1]

    out_model = MODELS_DIR / "vivino_clusterer_kmeans.joblib"
    out_csv = MODELS_DIR / "vivino_with_clusters.csv"

    joblib.dump(cluster_model, out_model)
    out_df.to_csv(out_csv, index=False, encoding="utf-8")

    print("\n=== КЛАСТЕРИЗАЦИЯ: итог ===")
    print("Модель сохранена:", out_model)
    print("CSV с кластерами сохранён:", out_csv)


if __name__ == "__main__":
    main()
