from __future__ import annotations

from functools import lru_cache
from typing import List, Optional, Tuple
import random

from .ttt_core import X, O, check_winner, available_moves, is_terminal, make_move


def _terminal_score(board: Tuple[int, ...]) -> int:
    w = check_winner(board)
    if w == X:
        return 1
    if w == O:
        return -1
    return 0


@lru_cache(maxsize=None)
def _minimax_full(board: Tuple[int, ...], player: int) -> int:
    """
    IDEAL teacher:
    full search until terminal.
    Score is from X perspective: +1 (X win), -1 (O win), 0 draw.
    """
    if is_terminal(board):
        return _terminal_score(board)

    moves = available_moves(board)
    if player == X:
        best = -2
        for m in moves:
            s = _minimax_full(make_move(board, m, X), O)
            best = max(best, s)
        return best
    else:
        best = 2
        for m in moves:
            s = _minimax_full(make_move(board, m, O), X)
            best = min(best, s)
        return best


@lru_cache(maxsize=None)
def _minimax_depth(board: Tuple[int, ...], player: int, depth: int) -> int:
    if is_terminal(board):
        return _terminal_score(board)
    if depth == 0:
        return 0

    moves = available_moves(board)
    if player == X:
        best = -2
        for m in moves:
            s = _minimax_depth(make_move(board, m, X), O, depth - 1)
            best = max(best, s)
        return best
    else:
        best = 2
        for m in moves:
            s = _minimax_depth(make_move(board, m, O), X, depth - 1)
            best = min(best, s)
        return best


def best_moves(board: Tuple[int, ...], player: int, depth: Optional[int]) -> List[int]:
    moves = available_moves(board)
    if not moves:
        return []

    scored = []
    if depth is None:
        for m in moves:
            s = _minimax_full(make_move(board, m, player), -player)
            scored.append((m, s))
    else:
        for m in moves:
            s = _minimax_depth(make_move(board, m, player), -player, max(depth - 1, 0))
            scored.append((m, s))

    if player == X:
        best_val = max(s for _, s in scored)
    else:
        best_val = min(s for _, s in scored)

    return [m for m, s in scored if s == best_val]


def teacher_move(board: Tuple[int, ...], player: int, level: str, rng: random.Random) -> int:
    """
    level:
      senior -> full minimax (ideal)
      mid    -> depth=3 + small noise
      junior -> depth=1 + bigger noise
    """
    if is_terminal(board):
        raise ValueError("Terminal position")

    moves = available_moves(board)
    if not moves:
        raise ValueError("No legal moves")

    if level == "senior":
        depth = None
        noise = 0.0
    elif level == "mid":
        depth = 3
        noise = 0.10
    elif level == "junior":
        depth = 1
        noise = 0.30
    else:
        raise ValueError("Unknown level")

    if rng.random() < noise:
        return rng.choice(moves)

    bm = best_moves(board, player, depth)
    return rng.choice(bm) if bm else rng.choice(moves)
