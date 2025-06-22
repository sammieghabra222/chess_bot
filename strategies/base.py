from typing import Protocol, Optional
import chess


class MoveStrategy(Protocol):
    """Anything that can choose a move given a board."""
    def select_move(self, board: chess.Board) -> Optional[chess.Move]:
        ...
