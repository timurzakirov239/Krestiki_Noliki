from __future__ import annotations

import sys
from pathlib import Path
import json
import random
import numpy as np
import joblib

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QGridLayout,
    QPushButton, QLabel, QComboBox, QGroupBox, QMessageBox,
    QSlider, QTextEdit
)

# --- ensure src is importable ---
BASE_DIR = Path(__file__).resolve().parents[2]  # project root
SRC_DIR = BASE_DIR / "src"
MODELS_DIR = BASE_DIR / "models"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from ttt.ttt_core import (  # noqa: E402
    X, O, EMPTY,
    check_winner, is_terminal, make_move,
    current_player, available_moves, encode_board_for_ai
)
from ttt.ttt_teacher import teacher_move  # noqa: E402
from ttt.ttt_agent import TTTAgent  # noqa: E402

from sklearn.neural_network import MLPClassifier  # noqa: E402


# ---------------- helpers ----------------
def cell_text(v: int) -> str:
    return "X" if v == X else ("O" if v == O else "")


def create_mlp(level: str, seed: int = 42) -> MLPClassifier:
    # configs for demo-training (supervised imitation)
    cfg = {
        "junior": dict(hidden=(16,), epochs=25, lr=0.08, alpha=1e-4),
        "mid": dict(hidden=(32, 32), epochs=45, lr=0.05, alpha=1e-4),
        "senior": dict(hidden=(64, 64), epochs=80, lr=0.03, alpha=1e-4),
    }[level]

    clf = MLPClassifier(
        hidden_layer_sizes=cfg["hidden"],
        activation="relu",
        solver="sgd",
        learning_rate_init=cfg["lr"],
        alpha=cfg["alpha"],
        momentum=0.9,
        nesterovs_momentum=True,
        max_iter=1,
        warm_start=False,
        random_state=seed,
    )
    return clf


# ---------------- Board widget ----------------
class BoardGrid(QWidget):
    def __init__(self, clickable: bool):
        super().__init__()
        self.clickable = clickable
        self._callbacks = {}

        layout = QGridLayout(self)
        layout.setSpacing(6)

        self.buttons: list[QPushButton] = []
        for i in range(9):
            btn = QPushButton("")
            btn.setFixedSize(78, 78)
            btn.setStyleSheet("font-size: 26px;")
            if clickable:
                btn.clicked.connect(lambda checked=False, idx=i: self._on_click(idx))
            else:
                btn.setEnabled(False)
            self.buttons.append(btn)
            layout.addWidget(btn, i // 3, i % 3)

        self.reset_styles()

    def on_cell(self, idx: int, cb):
        self._callbacks[idx] = cb

    def _on_click(self, idx: int):
        cb = self._callbacks.get(idx)
        if cb:
            cb(idx)

    def set_board(self, board: tuple[int, ...]):
        for i, b in enumerate(self.buttons):
            b.setText(cell_text(board[i]))

    def reset_styles(self):
        for b in self.buttons:
            b.setStyleSheet("font-size: 26px; border: 1px solid #999;")

    def set_highlights(self, teacher_idx: int | None, model_idx: int | None, illegal_model: bool = False):
        self.reset_styles()

        # teacher highlight (green)
        if teacher_idx is not None:
            self.buttons[teacher_idx].setStyleSheet(
                "font-size: 26px; border: 3px solid #2e7d32;"
            )

        # model highlight (orange / red if illegal)
        if model_idx is not None:
            color = "#c62828" if illegal_model else "#ef6c00"
            self.buttons[model_idx].setStyleSheet(
                f"font-size: 26px; border: 3px solid {color};"
            )

        # if same — make it blue
        if teacher_idx is not None and model_idx is not None and teacher_idx == model_idx:
            self.buttons[teacher_idx].setStyleSheet(
                "font-size: 26px; border: 3px solid #1565c0;"
            )


# ---------------- Training demo (slow motion) ----------------
class SlowSupervisedTrainer(QWidget):
    """
    Показывает обучение с учителем "в замедленном виде":
    - на каждом шаге учитель (minimax) говорит правильный ход
    - модель предсказывает свой ход
    - модель обновляется (partial_fit) на метке учителя
    - дальше игра продолжается (модель vs учитель), чтобы было видно все ходы
    """

    def __init__(self):
        super().__init__()

        self.rng = random.Random(42)

        self.teacher_level = "senior"
        self.model_level = "junior"
        self.model_side = X  # model plays X by default

        self.model: MLPClassifier | None = None
        self.model_fitted = False
        self.classes = np.arange(9, dtype=np.int64)

        self.board = tuple([EMPTY] * 9)
        self.games = 0
        self.steps = 0
        self.correct = 0

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.step)

        root = QVBoxLayout(self)

        # controls
        controls = QHBoxLayout()
        root.addLayout(controls)

        self.cmb_teacher = QComboBox()
        self.cmb_teacher.addItems(["junior", "mid", "senior"])
        self.cmb_teacher.setCurrentText("senior")
        self.cmb_teacher.currentTextChanged.connect(self._on_teacher_changed)

        self.cmb_model = QComboBox()
        self.cmb_model.addItems(["junior", "mid", "senior"])
        self.cmb_model.setCurrentText("junior")
        self.cmb_model.currentTextChanged.connect(self._on_model_changed)

        self.cmb_side = QComboBox()
        self.cmb_side.addItems(["Модель играет за X", "Модель играет за O"])
        self.cmb_side.currentIndexChanged.connect(self._on_side_changed)

        self.btn_reset = QPushButton("Сброс")
        self.btn_reset.clicked.connect(self.reset)

        self.btn_start = QPushButton("Старт")
        self.btn_start.clicked.connect(self.toggle)

        self.btn_step = QPushButton("Шаг")
        self.btn_step.clicked.connect(self.step)

        controls.addWidget(QLabel("Учитель:"))
        controls.addWidget(self.cmb_teacher)
        controls.addWidget(QLabel("Модель:"))
        controls.addWidget(self.cmb_model)
        controls.addWidget(self.cmb_side)
        controls.addStretch(1)
        controls.addWidget(self.btn_reset)
        controls.addWidget(self.btn_start)
        controls.addWidget(self.btn_step)

        # speed slider
        speed_row = QHBoxLayout()
        root.addLayout(speed_row)

        self.lbl_speed = QLabel("Скорость: 350 мс/шаг")
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(60, 1200)
        self.slider.setValue(350)
        self.slider.valueChanged.connect(self._on_speed_changed)

        speed_row.addWidget(self.lbl_speed)
        speed_row.addWidget(self.slider)

        # info + board
        self.lbl_info = QLabel("Готово. Нажмите «Старт» — увидите как модель учится у учителя.")
        self.lbl_info.setWordWrap(True)
        root.addWidget(self.lbl_info)

        self.board_view = BoardGrid(clickable=False)
        root.addWidget(self.board_view)

        # legend
        self.lbl_legend = QLabel(
            "Легенда: зелёная рамка — ход учителя, оранжевая — ход модели, синяя — совпало, красная — модель выбрала занятое (плохое)."
        )
        self.lbl_legend.setWordWrap(True)
        root.addWidget(self.lbl_legend)

        # log
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(140)
        root.addWidget(self.log)

        self.reset()

    def _on_teacher_changed(self, t: str):
        self.teacher_level = t

    def _on_model_changed(self, t: str):
        self.model_level = t
        self.reset_model_only()

    def _on_side_changed(self, idx: int):
        self.model_side = X if idx == 0 else O
        self.reset()

    def _on_speed_changed(self, v: int):
        self.lbl_speed.setText(f"Скорость: {v} мс/шаг")
        if self.timer.isActive():
            self.timer.setInterval(v)

    def reset_model_only(self):
        self.model = create_mlp(self.model_level, seed=42)
        self.model_fitted = False

    def reset(self):
        self.timer.stop()
        self.btn_start.setText("Старт")

        self.reset_model_only()
        self.board = tuple([EMPTY] * 9)

        self.games = 0
        self.steps = 0
        self.correct = 0

        self.board_view.set_board(self.board)
        self.board_view.set_highlights(None, None)
        self.log.clear()

        self._update_info("Сброшено. Нажмите «Старт».")

    def toggle(self):
        if self.timer.isActive():
            self.timer.stop()
            self.btn_start.setText("Старт")
        else:
            self.timer.start(self.slider.value())
            self.btn_start.setText("Пауза")

    def _update_info(self, extra: str = ""):
        acc = (self.correct / self.steps * 100.0) if self.steps > 0 else 0.0
        self.lbl_info.setText(
            f"Игры: {self.games} | Шаги обучения: {self.steps} | Совпадений с учителем: {acc:.1f}%\n{extra}"
        )

    def _append_log(self, s: str):
        self.log.append(s)
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    def _predict_model_move(self, board: tuple[int, ...], player_to_move: int) -> tuple[int, bool]:
        moves = available_moves(board)
        if not moves:
            return 0, False

        # before fitted => random
        if not self.model_fitted or self.model is None:
            return random.choice(moves), False

        x_vec = encode_board_for_ai(board, player_to_move)
        X_in = np.array([x_vec], dtype=np.float32)
        proba = self.model.predict_proba(X_in)[0].astype(float)

        # choose argmax (without masking first to show "illegal" potential)
        raw_choice = int(np.argmax(proba))
        illegal = raw_choice not in moves

        # for actual execution we must ensure legal => mask
        if illegal:
            proba_masked = proba.copy()
            for i in range(9):
                if i not in moves:
                    proba_masked[i] = -1e9
            legal_choice = int(np.argmax(proba_masked))
            return legal_choice, True

        return raw_choice, False

    def step(self):
        # start new game if terminal
        if is_terminal(self.board) or len(available_moves(self.board)) == 0:
            self.games += 1
            self.board = tuple([EMPTY] * 9)

        player = current_player(self.board)

        # teacher chooses best move for current player
        t_move = teacher_move(self.board, player, self.teacher_level, self.rng)

        if player == self.model_side:
            # model prediction (and train step)
            m_move, had_illegal = self._predict_model_move(self.board, player)

            # training update: model learns teacher's move for this state
            x_vec = encode_board_for_ai(self.board, player)
            X_in = np.array([x_vec], dtype=np.float32)
            y = np.array([t_move], dtype=np.int64)

            if self.model is None:
                self.reset_model_only()

            if not self.model_fitted:
                self.model.partial_fit(X_in, y, classes=self.classes)
                self.model_fitted = True
            else:
                self.model.partial_fit(X_in, y)

            self.steps += 1
            if m_move == t_move:
                self.correct += 1

            # visualize
            self.board_view.set_board(self.board)
            self.board_view.set_highlights(t_move, m_move, illegal_model=had_illegal)

            self._append_log(
                f"Ход модели ({'X' if player == X else 'O'}): модель={m_move}, учитель={t_move} | "
                f"{'совпало' if m_move == t_move else 'не совпало'}"
            )

            # execute: модель ходит (модель vs учитель)
            try:
                self.board = make_move(self.board, m_move, player)
            except Exception:
                # fallback to teacher if something goes wrong
                self.board = make_move(self.board, t_move, player)

        else:
            # opponent turn: teacher plays
            self.board_view.set_board(self.board)
            self.board_view.set_highlights(t_move, None)
            self._append_log(f"Ход учителя ({'X' if player == X else 'O'}): {t_move}")
            self.board = make_move(self.board, t_move, player)

        # finish check
        w = check_winner(self.board)
        if w != 0:
            self._append_log(f"Игра закончилась: победили {'X' if w == X else 'O'}")
        elif is_terminal(self.board):
            self._append_log("Игра закончилась: ничья")

        self._update_info()

    def save_current_model(self):
        if self.model is None or not self.model_fitted:
            QMessageBox.information(self, "Сохранение", "Модель ещё не обучалась.")
            return

        level = self.model_level
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / f"ttt_{level}.joblib"
        meta_path = MODELS_DIR / f"ttt_{level}_meta.json"

        joblib.dump(self.model, model_path)
        meta = {
            "level": level,
            "teacher": self.teacher_level,
            "steps": int(self.steps),
            "games": int(self.games),
            "match_teacher_percent": float((self.correct / self.steps * 100.0) if self.steps else 0.0),
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        QMessageBox.information(self, "Сохранение", f"Сохранено:\n{model_path}")


# ---------------- Left: play widget ----------------
class PlayTicTacToe(QWidget):
    def __init__(self):
        super().__init__()
        self.agent = TTTAgent(seed=42)
        self.board = tuple([EMPTY] * 9)
        self.game_over = False

        self.user_mark = X
        self.ai_mark = O

        root = QVBoxLayout(self)

        top = QHBoxLayout()
        root.addLayout(top)

        self.cmb_level = QComboBox()
        self.cmb_level.addItems(["junior", "mid", "senior"])

        self.cmb_side = QComboBox()
        self.cmb_side.addItems(["Игрок: X", "Игрок: O"])
        self.cmb_side.currentIndexChanged.connect(self.on_side_changed)

        self.btn_new = QPushButton("Новая игра")
        self.btn_new.clicked.connect(self.new_game)

        top.addWidget(QLabel("Модель:"))
        top.addWidget(self.cmb_level)
        top.addWidget(self.cmb_side)
        top.addStretch(1)
        top.addWidget(self.btn_new)

        self.status = QLabel("Ваш ход")
        root.addWidget(self.status)

        self.grid = BoardGrid(clickable=True)
        for i in range(9):
            self.grid.on_cell(i, self.on_cell_clicked)
        root.addWidget(self.grid)

        self.update_ui()
        self.maybe_ai_move()

    def on_side_changed(self, idx: int):
        self.user_mark = X if idx == 0 else O
        self.ai_mark = O if idx == 0 else X
        self.new_game()

    def new_game(self):
        self.board = tuple([EMPTY] * 9)
        self.game_over = False
        self.status.setText("Новая игра.")
        self.update_ui()
        self.maybe_ai_move()

    def update_ui(self):
        self.grid.set_board(self.board)
        self.grid.set_highlights(None, None)
        for i, btn in enumerate(self.grid.buttons):
            btn.setEnabled((not self.game_over) and (self.board[i] == EMPTY))

    def on_cell_clicked(self, idx: int):
        if self.game_over or self.board[idx] != EMPTY:
            return
        if current_player(self.board) != self.user_mark:
            return

        self.board = make_move(self.board, idx, self.user_mark)
        self.after_move()
        if not self.game_over:
            self.maybe_ai_move()

    def maybe_ai_move(self):
        if self.game_over:
            return
        if current_player(self.board) != self.ai_mark:
            return

        level = self.cmb_level.currentText()
        try:
            move = self.agent.choose_move(self.board, self.ai_mark, level)
            self.board = make_move(self.board, move, self.ai_mark)
            self.after_move()
        except Exception as e:
            QMessageBox.warning(self, "Ошибка", str(e))

    def after_move(self):
        w = check_winner(self.board)
        if w == X:
            self.game_over = True
            self.status.setText("Победили X!")
        elif w == O:
            self.game_over = True
            self.status.setText("Победили O!")
        elif is_terminal(self.board):
            self.game_over = True
            self.status.setText("Ничья.")
        else:
            self.status.setText("Игра продолжается.")
        self.update_ui()


# ---------------- Home widget (game + training demo) ----------------
class TicTacToeHomeWidget(QWidget):
    def __init__(self):
        super().__init__()
        root = QHBoxLayout(self)

        # Left: game
        gb_game = QGroupBox("Игра")
        lay_game = QVBoxLayout(gb_game)
        self.play = PlayTicTacToe()
        lay_game.addWidget(self.play)
        root.addWidget(gb_game, 2)

        # Right: slow training
        gb_train = QGroupBox("Обучение (замедленно)")
        lay_train = QVBoxLayout(gb_train)
        self.trainer = SlowSupervisedTrainer()

        # add save button under trainer controls
        btn_save = QPushButton("Сохранить текущую модель (в models/)")
        btn_save.clicked.connect(self.trainer.save_current_model)
        lay_train.addWidget(self.trainer)
        lay_train.addWidget(btn_save)

        root.addWidget(gb_train, 3)
