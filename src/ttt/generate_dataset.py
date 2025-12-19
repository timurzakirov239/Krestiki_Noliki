from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path
import random
import numpy as np

from .ttt_core import (
    X, O, EMPTY,
    is_terminal, is_valid_reachable, available_moves,
    current_player, encode_board_for_ai,
    TRANSFORMS, INVERSE_TRANSFORMS,
    apply_transform, transform_move
)
from .ttt_teacher import teacher_move


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "ttt"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def generate(level: str, seed: int = 42, augment: bool = True):
    rng = random.Random(seed)
    X_list = []
    y_list = []

    for cells in product([O, EMPTY, X], repeat=9):
        board = tuple(cells)

        if not is_valid_reachable(board):
            continue
        if is_terminal(board):
            continue

        player = current_player(board)
        if not available_moves(board):
            continue

        move = teacher_move(board, player, level, rng)

        # encode from "AI perspective" = current player to move
        x_vec = encode_board_for_ai(board, player)

        if augment:
            for mapping, inv in zip(TRANSFORMS, INVERSE_TRANSFORMS):
                b2 = apply_transform(tuple(x_vec), mapping)  # safe: x_vec is also 9 ints
                m2 = transform_move(move, inv)
                X_list.append(list(b2))
                y_list.append(m2)
        else:
            X_list.append(x_vec)
            y_list.append(move)

    X_arr = np.array(X_list, dtype=np.float32)
    y_arr = np.array(y_list, dtype=np.int64)
    return X_arr, y_arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", choices=["junior", "mid", "senior"], required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-augment", action="store_true")
    args = ap.parse_args()

    X_arr, y_arr = generate(args.level, seed=args.seed, augment=(not args.no_augment))
    out_path = DATA_DIR / f"ttt_{args.level}.npz"
    np.savez_compressed(out_path, X=X_arr, y=y_arr)

    print("\n=== Датасет крестики-нолики (с учителем) ===")
    print(f"Уровень: {args.level}")
    print(f"Размер: X={X_arr.shape}, y={y_arr.shape}")
    print("Файл сохранён:", out_path)


if __name__ == "__main__":
    main()
