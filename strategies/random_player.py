import random, chess
from typing import Optional
from .base import MoveStrategy


class RandomPlayer(MoveStrategy):
    def select_move(self, board: chess.Board) -> Optional[chess.Move]:
        moves = list(board.legal_moves)
        return random.choice(moves) if moves else None
