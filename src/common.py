from __future__ import annotations

from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"


WINE_FILES = [
    ("red", "Red.csv"),
    ("white", "White.csv"),
    ("rose", "Rose.csv"),
    ("sparkling", "Sparkling.csv"),
]


def load_vivino() -> pd.DataFrame:
    frames = []
    for wine_type, fname in WINE_FILES:
        path = DATA_DIR / fname
        if not path.exists():
            raise FileNotFoundError(f"Не найден файл: {path}")
        df = pd.read_csv(path)

        # нормализуем колонки (на всякий)
        df.columns = [c.strip() for c in df.columns]

        df["WineType"] = wine_type
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)

    # ожидаемые колонки
    expected = {"Name", "Country", "Region", "Winery", "Rating", "NumberOfRatings", "Price", "Year", "WineType"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"В данных нет колонок: {missing}. Колонки есть: {list(df.columns)}")

    # типы
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df["NumberOfRatings"] = pd.to_numeric(df["NumberOfRatings"], errors="coerce")

    # Year: есть N.V.
    df["Year_is_nv"] = df["Year"].astype(str).str.upper().str.contains("N.V")
    df["Year_num"] = pd.to_numeric(df["Year"], errors="coerce")  # N.V. -> NaN

    # чистим откровенно битые строки
    df = df.dropna(subset=["Price", "Rating", "NumberOfRatings"])
    df = df[df["Price"] > 0].copy()
    df = df[df["NumberOfRatings"] > 0].copy()

    # заполняем Year_num медианой (можно будет улучшить)
    df["Year_num"] = df["Year_num"].fillna(df["Year_num"].median())

    return df


def make_features(df: pd.DataFrame, use_name_text: bool = False):
    """
    Возвращает (X, numeric_cols, cat_cols, text_col_or_None)
    """
    numeric_cols = ["Rating", "NumberOfRatings", "Year_num", "Year_is_nv"]
    cat_cols = ["Country", "Region", "Winery", "WineType"]
    text_col = "Name" if use_name_text else None

    # X включает всё, что нужно препроцессору
    cols = numeric_cols + cat_cols + ([text_col] if text_col else [])
    X = df[cols].copy()

    # Year_is_nv -> 0/1
    X["Year_is_nv"] = X["Year_is_nv"].astype(int)

    return X, numeric_cols, cat_cols, text_col
