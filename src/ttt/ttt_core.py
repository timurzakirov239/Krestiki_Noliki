from __future__ import annotations

from typing import List, Tuple

X = 1
O = -1
EMPTY = 0

WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6),
]


def check_winner(board: Tuple[int, ...]) -> int:
    for a, b, c in WIN_LINES:
        s = board[a] + board[b] + board[c]
        if s == 3:
            return X
        if s == -3:
            return O
    return 0


def available_moves(board: Tuple[int, ...]) -> List[int]:
    return [i for i, v in enumerate(board) if v == EMPTY]


def is_terminal(board: Tuple[int, ...]) -> bool:
    if check_winner(board) != 0:
        return True
    return all(v != EMPTY for v in board)


def make_move(board: Tuple[int, ...], idx: int, player: int) -> Tuple[int, ...]:
    if board[idx] != EMPTY:
        raise ValueError("Illegal move: cell is not empty")
    b = list(board)
    b[idx] = player
    return tuple(b)


def current_player(board: Tuple[int, ...]) -> int:
    cx = sum(1 for v in board if v == X)
    co = sum(1 for v in board if v == O)
    return X if cx == co else O


def is_valid_reachable(board: Tuple[int, ...]) -> bool:
    cx = sum(1 for v in board if v == X)
    co = sum(1 for v in board if v == O)

    if not (cx == co or cx == co + 1):
        return False

    w = check_winner(board)
    if w == 0:
        return True

    if w == X and cx != co + 1:
        return False
    if w == O and cx != co:
        return False

    return True


def encode_board_for_ai(board: Tuple[int, ...], ai_player: int) -> List[int]:
    """
    1 = AI marks, -1 = opponent marks, 0 = empty.
    If AI is O, flip signs so AI becomes +1 in representation.
    """
    if ai_player == X:
        return list(board)
    return [(-v) for v in board]


# ---- Symmetries (augmentation) ----
TRANSFORMS = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8],
    [6, 3, 0, 7, 4, 1, 8, 5, 2],
    [8, 7, 6, 5, 4, 3, 2, 1, 0],
    [2, 5, 8, 1, 4, 7, 0, 3, 6],
    [2, 1, 0, 5, 4, 3, 8, 7, 6],
    [6, 7, 8, 3, 4, 5, 0, 1, 2],
    [0, 3, 6, 1, 4, 7, 2, 5, 8],
    [8, 5, 2, 7, 4, 1, 6, 3, 0],
]


def inverse_mapping(mapping: List[int]) -> List[int]:
    inv = [0] * 9
    for new_i, old_i in enumerate(mapping):
        inv[old_i] = new_i
    return inv


INVERSE_TRANSFORMS = [inverse_mapping(m) for m in TRANSFORMS]


def apply_transform(board: Tuple[int, ...], mapping: List[int]) -> Tuple[int, ...]:
    return tuple(board[mapping[i]] for i in range(9))


def transform_move(move_idx: int, inv_mapping: List[int]) -> int:
    return inv_mapping[move_idx]
