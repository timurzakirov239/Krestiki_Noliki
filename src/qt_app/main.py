from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget,
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QMessageBox, QTableView, QGroupBox, QFormLayout
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.lines import Line2D


# --- paths ---
BASE_DIR = Path(__file__).resolve().parents[2]  # project root
MODELS_DIR = BASE_DIR / "models"
DEFAULT_DATA_CSV = MODELS_DIR / "vivino_with_clusters.csv"

# allow importing src/common.py
sys.path.append(str(BASE_DIR / "src"))
from common import make_features  # noqa: E402


# --- Column names (display only) ---
COLUMN_NAME_RU = {
    "Name": "Имя",
    "Country": "Страна",
    "Region": "Регион",
    "Winery": "Винодельня",
    "Rating": "Рейтинг",
    "NumberOfRatings": "Число оценок",
    "Price": "Цена (€)",
    "Year": "Год (как в данных)",
    "WineType": "Тип вина",
    "Year_is_nv": "Без винтажа (N.V.)",
    "Year_num": "Год (число)",
    "cluster": "Кластер",
    "x2d": "Проекция 1",
    "y2d": "Проекция 2",
}


# ---------- Pandas table model ----------
class PandasModel(QAbstractTableModel):
    def __init__(self, df: pd.DataFrame, header_map: dict[str, str] | None = None):
        super().__init__()
        self._df = df
        self._header_map = header_map or {}

    def rowCount(self, parent=QModelIndex()):
        return len(self._df)

    def columnCount(self, parent=QModelIndex()):
        return len(self._df.columns)

    def data(self, index: QModelIndex, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        if role == Qt.DisplayRole:
            val = self._df.iat[index.row(), index.column()]
            if isinstance(val, float):
                return f"{val:.4f}".rstrip("0").rstrip(".")
            return str(val)
        return None

    def headerData(self, section: int, orientation: Qt.Orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            col = str(self._df.columns[section])
            return self._header_map.get(col, col)
        return str(section)

    def set_df(self, df: pd.DataFrame):
        self.beginResetModel()
        self._df = df
        self.endResetModel()

    @property
    def df(self) -> pd.DataFrame:
        return self._df


def price_to_class_ru(price: float, q33: float, q66: float) -> str:
    if price <= q33:
        return "дёшево"
    if price <= q66:
        return "средне"
    return "дорого"


# ---------- Main window ----------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vivino ML")

        self.df: pd.DataFrame | None = None

        # models
        self.reg_model = None
        self.cls_model = None
        self.thresholds = None
        self.cluster_names: dict[int, str] = {}

        self._load_models()

        # Top-level tabs: only Home + Data
        self.tabs = QTabWidget()
        self.tab_home = self._build_tab_home_empty()
        self.tab_data_container = self._build_tab_data_container()

        self.tabs.addTab(self.tab_home, "Главная")
        self.tabs.addTab(self.tab_data_container, "Данные")
        self.setCentralWidget(self.tabs)

        # Auto-load default dataset if exists
        if DEFAULT_DATA_CSV.exists():
            self.load_csv(DEFAULT_DATA_CSV)

    # ---- models ----
    def _load_models(self):
        try:
            reg_path = MODELS_DIR / "vivino_regressor_price.joblib"
            cls_path = MODELS_DIR / "vivino_classifier_priceclass.joblib"
            thr_path = MODELS_DIR / "vivino_priceclass_thresholds.json"

            self.reg_model = joblib.load(reg_path) if reg_path.exists() else None
            self.cls_model = joblib.load(cls_path) if cls_path.exists() else None
            self.thresholds = json.loads(thr_path.read_text(encoding="utf-8")) if thr_path.exists() else None
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить модели:\n{e}")

    # ---- Home (empty) ----
    def _build_tab_home_empty(self) -> QWidget:
        # Intentionally empty: user will add a game later
        return QWidget()

    # ---- Data container (subtabs) ----
    def _build_tab_data_container(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        top = QHBoxLayout()
        self.lbl_path = QLabel("Файл не загружен")
        btn_open = QPushButton("Загрузить CSV...")
        btn_open.clicked.connect(self.on_open_csv)

        top.addWidget(self.lbl_path)
        top.addStretch(1)
        top.addWidget(btn_open)
        layout.addLayout(top)

        self.data_tabs = QTabWidget()
        self.tab_table = self._build_subtab_table()
        self.tab_reg = self._build_subtab_regression()
        self.tab_cls = self._build_subtab_classification()
        self.tab_clu = self._build_subtab_clusters()

        self.data_tabs.addTab(self.tab_table, "Таблица")
        self.data_tabs.addTab(self.tab_reg, "Регрессия")
        self.data_tabs.addTab(self.tab_cls, "Классификация")
        self.data_tabs.addTab(self.tab_clu, "Кластеры")

        layout.addWidget(self.data_tabs)
        return w

    # ---- load csv ----
    def on_open_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите CSV", str(BASE_DIR), "CSV files (*.csv)"
        )
        if path:
            self.load_csv(Path(path))

    def _normalize_loaded_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [c.strip() for c in df.columns]

        for col in ["Rating", "Price", "NumberOfRatings"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "Year" in df.columns and "Year_is_nv" not in df.columns:
            df["Year_is_nv"] = df["Year"].astype(str).str.upper().str.contains("N.V")

        if "Year" in df.columns and "Year_num" not in df.columns:
            df["Year_num"] = pd.to_numeric(df["Year"], errors="coerce")

        if "Year_num" in df.columns:
            if df["Year_num"].isna().all():
                df["Year_num"] = 0
            else:
                df["Year_num"] = df["Year_num"].fillna(df["Year_num"].median())

        if "Year_is_nv" in df.columns:
            df["Year_is_nv"] = df["Year_is_nv"].fillna(False)

        return df

    def load_csv(self, path: Path):
        try:
            df = pd.read_csv(path)
            df = self._normalize_loaded_df(df)
            self.df = df

            self.table_model.set_df(df)
            self.lbl_path.setText(str(path))

            self._refresh_cluster_names()
            self._refresh_cluster_plot()
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить CSV:\n{e}")

    # ---- selection helpers ----
    def _selected_row_index(self) -> int | None:
        if self.df is None or len(self.df) == 0:
            return None
        sel = self.table.selectionModel()
        if sel is None or not sel.hasSelection():
            return None
        return sel.selectedRows()[0].row()

    def _ensure_data_selected(self) -> int | None:
        if self.df is None:
            QMessageBox.warning(self, "Нет данных", "Сначала загрузите CSV.")
            return None
        row_idx = self._selected_row_index()
        if row_idx is None:
            QMessageBox.information(self, "Выбор строки", "Выберите строку в таблице.")
            return None
        return row_idx

    def _build_X_from_row(self, row: pd.Series) -> pd.DataFrame:
        one_df = pd.DataFrame([row])
        X_row, _, _, _ = make_features(one_df, use_name_text=False)
        return X_row

    # ---- Subtab: Table ----
    def _build_subtab_table(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        self.table_model = PandasModel(pd.DataFrame(), header_map=COLUMN_NAME_RU)
        self.table = QTableView()
        self.table.setModel(self.table_model)
        self.table.setSelectionBehavior(QTableView.SelectRows)
        self.table.setSelectionMode(QTableView.SingleSelection)

        layout.addWidget(self.table)
        return w

    # ---- Subtab: Regression ----
    def _build_subtab_regression(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        btn = QPushButton("Предсказать цену по выбранной строке")
        btn.clicked.connect(self.on_predict_regression)
        layout.addWidget(btn)

        box = QGroupBox("Результаты")
        form = QFormLayout(box)

        self.reg_out_name = QLabel("-")
        self.reg_out_true = QLabel("-")
        self.reg_out_pred = QLabel("-")

        form.addRow("Имя:", self.reg_out_name)
        form.addRow("Цена (в данных):", self.reg_out_true)
        form.addRow("Цена (прогноз):", self.reg_out_pred)

        layout.addWidget(box)
        layout.addStretch(1)
        return w

    def on_predict_regression(self):
        if self.reg_model is None:
            QMessageBox.warning(self, "Нет модели", "Не найдена модель регрессии в папке models/.")
            return
        row_idx = self._ensure_data_selected()
        if row_idx is None:
            return

        row = self.df.iloc[row_idx].copy()
        self.reg_out_name.setText(str(row.get("Name", "-")))

        true_price = row.get("Price", np.nan)
        self.reg_out_true.setText(f"{float(true_price):.2f} €" if pd.notna(true_price) else "-")

        try:
            X_row = self._build_X_from_row(row)
            pred_log = float(self.reg_model.predict(X_row)[0])
            pred_price = float(np.expm1(pred_log))
            self.reg_out_pred.setText(f"{pred_price:.2f} €")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сделать предсказание:\n{e}")

    # ---- Subtab: Classification ----
    def _build_subtab_classification(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        btn = QPushButton("Предсказать ценовой сегмент по выбранной строке")
        btn.clicked.connect(self.on_predict_classification)
        layout.addWidget(btn)

        box = QGroupBox("Результаты")
        form = QFormLayout(box)

        self.cls_out_name = QLabel("-")
        self.cls_out_true = QLabel("-")
        self.cls_out_pred = QLabel("-")

        form.addRow("Имя:", self.cls_out_name)
        form.addRow("Сегмент (по цене):", self.cls_out_true)
        form.addRow("Сегмент (прогноз):", self.cls_out_pred)

        layout.addWidget(box)
        layout.addStretch(1)
        return w

    def on_predict_classification(self):
        if self.cls_model is None:
            QMessageBox.warning(self, "Нет модели", "Не найдена модель классификации в папке models/.")
            return
        if self.thresholds is None:
            QMessageBox.warning(self, "Нет порогов", "Не найден файл порогов vivino_priceclass_thresholds.json.")
            return

        row_idx = self._ensure_data_selected()
        if row_idx is None:
            return

        row = self.df.iloc[row_idx].copy()
        self.cls_out_name.setText(str(row.get("Name", "-")))

        q33 = float(self.thresholds.get("q33"))
        q66 = float(self.thresholds.get("q66"))

        true_price = row.get("Price", np.nan)
        if pd.notna(true_price):
            self.cls_out_true.setText(price_to_class_ru(float(true_price), q33, q66))
        else:
            self.cls_out_true.setText("-")

        try:
            X_row = self._build_X_from_row(row)
            pred_class = str(self.cls_model.predict(X_row)[0])
            self.cls_out_pred.setText(pred_class)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сделать предсказание:\n{e}")

    # ---- Subtab: Clusters ----
    def _build_subtab_clusters(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        self.fig = Figure(figsize=(7, 5))
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)

        layout.addWidget(self.canvas)
        layout.addStretch(1)
        return w

    def _refresh_cluster_names(self):
        """
        Даём кластерам названия по их распределению (по средним значениям в каждом кластере).
        """
        self.cluster_names = {}
        if self.df is None or "cluster" not in self.df.columns:
            return

        df = self.df.copy()
        df = df[pd.notna(df["cluster"])].copy()
        if df.empty:
            return

        df["cluster"] = df["cluster"].astype(int)
        grp = df.groupby("cluster", sort=True)

        summary = pd.DataFrame({
            "cluster": grp.size().index.astype(int),
            "count": grp.size().values,
            "avg_price": grp["Price"].mean().values if "Price" in df.columns else np.nan,
            "avg_rating": grp["Rating"].mean().values if "Rating" in df.columns else np.nan,
            "avg_pop": grp["NumberOfRatings"].mean().values if "NumberOfRatings" in df.columns else np.nan,
            "type": grp["WineType"].agg(lambda s: s.value_counts().index[0]).values if "WineType" in df.columns else "",
        })

        med_price = float(pd.Series(summary["avg_price"]).median())
        med_rating = float(pd.Series(summary["avg_rating"]).median())
        med_pop = float(pd.Series(summary["avg_pop"]).median())

        for _, r in summary.iterrows():
            cid = int(r["cluster"])
            price_hi = float(r["avg_price"]) >= med_price
            rating_hi = float(r["avg_rating"]) >= med_rating
            pop_hi = float(r["avg_pop"]) >= med_pop

            if price_hi and rating_hi:
                base = "премиальные, высокие оценки"
            elif (not price_hi) and rating_hi:
                base = "выгодные, высокие оценки"
            elif price_hi and (not rating_hi):
                base = "дорогие, спорные оценки"
            else:
                base = "бюджетные, ниже среднего"

            pop = "популярные" if pop_hi else "нишевые"

            wt = str(r.get("type", "")).strip()
            ru_type = {"red": "красные", "white": "белые", "rose": "розовые", "sparkling": "игристые"}.get(wt, wt)
            tail = f", {ru_type}" if ru_type else ""

            self.cluster_names[cid] = f"{base}, {pop}{tail}"

    def _refresh_cluster_plot(self):
        self.ax.clear()

        if self.df is None:
            self.ax.set_title("Нет данных")
            self.canvas.draw()
            return

        needed = {"x2d", "y2d", "cluster"}
        if not needed.issubset(set(self.df.columns)):
            self.ax.set_title("Для графика нужен файл vivino_with_clusters.csv (x2d, y2d, cluster)")
            self.canvas.draw()
            return

        x = self.df["x2d"].values
        y = self.df["y2d"].values
        c_raw = self.df["cluster"].values

        mask = ~pd.isna(c_raw)
        x = x[mask]
        y = y[mask]
        c = c_raw[mask].astype(int)

        unique_clusters = np.unique(c)
        k = len(unique_clusters)
        cmap_name = "tab10" if k <= 10 else "tab20"

        sc = self.ax.scatter(x, y, c=c, s=8, cmap=cmap_name)
        self.ax.set_title("Кластеры вин")
        self.ax.set_xlabel("Проекция 1")
        self.ax.set_ylabel("Проекция 2")

        # Legend: cluster -> name -> count
        counts = pd.Series(c).value_counts().sort_index()
        handles = []
        labels = []
        colormap = sc.cmap
        norm = sc.norm

        for cid, cnt in counts.items():
            color = colormap(norm(cid))
            name = self.cluster_names.get(int(cid), f"Кластер {int(cid)}")
            handles.append(Line2D([0], [0], marker="o", linestyle="None",markerfacecolor=color, markeredgecolor=color, markersize=7))
            labels.append(f"{int(cid)}: {name} ({int(cnt)})")

        self.ax.legend(handles, labels, loc="upper right", fontsize=8, frameon=True)
        self.canvas.draw()


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(1250, 750)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
