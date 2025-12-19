from __future__ import annotations

from pathlib import Path
import random
import numpy as np
import joblib

from .ttt_core import X, O, available_moves, is_terminal, encode_board_for_ai
from .ttt_teacher import teacher_move


BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "models"

EPSILON = {"junior": 0.30, "mid": 0.10, "senior": 0.00}


class TTTAgent:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.models = {}

    def _load_model(self, level: str):
        if level in self.models:
            return self.models[level]
        path = MODELS_DIR / f"ttt_{level}.joblib"
        self.models[level] = joblib.load(path) if path.exists() else None
        return self.models[level]

    def choose_move(self, board: tuple[int, ...], ai_player: int, level: str) -> int:
        if is_terminal(board):
            raise ValueError("Игра уже завершена")

        moves = available_moves(board)
        if not moves:
            raise ValueError("Нет доступных ходов")

        if self.rng.random() < EPSILON[level]:
            return self.rng.choice(moves)

        model = self._load_model(level)
        if model is None:
            return teacher_move(board, ai_player, level, self.rng)

        x_vec = encode_board_for_ai(board, ai_player)
        X_in = np.array([x_vec], dtype=np.float32)

        proba = model.predict_proba(X_in)[0].astype(float)

        # mask illegal moves
        for i in range(9):
            if i not in moves:
                proba[i] = -1e9

        return int(np.argmax(proba))
