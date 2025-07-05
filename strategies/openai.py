import chess
import random
from collections import namedtuple
from chess import polyglot

tt_entry = namedtuple("TTEntry", "depth flag score best_move")
EXACT, LOWERBOUND, UPPERBOUND = 0, 1, 2

class OpenAIStrategy:
    """
    A MoveStrategy implementation that selects moves using minimax search
    with alpha-beta pruning and heuristic evaluation. It now includes an
    expanded, randomized opening book.
    """

    # Expanded opening repertoire with multiple variations
    OPENINGS = {
        "Sicilian Defense": [
            ["e4", "c5"]
        ],
        "Ruy Lopez": [
            ["e4", "e5", "Nf3", "Nc6", "Bb5"]
        ],
        "Italian Game": [
            ["e4", "e5", "Nf3", "Nc6", "Bc4"]
        ],
        "French Defense": [
            ["e4", "e6"]
        ],
        "Caro-Kann Defense": [
            ["e4", "c6"]
        ],
        "Queen's Gambit": [
            ["d4", "d5", "c4"]
        ],
        "King's Indian Defense": [
            ["d4", "Nf6", "c4", "g6"]
        ],
        "Slav Defense": [
            ["d4", "d5", "c4", "c6"]
        ],
        "Scandinavian Defense": [
            ["e4", "d5"]
        ],
        "Double King's Pawn": [
            ["e4", "e5"]
        ]
    }

    def __init__(self, depth: int = 5):
        self.depth = depth  # search depth in plies (half-moves)

        # --- MODIFICATION: Randomly select an opening ---
        # A random opening name is chosen from the dictionary keys
        opening_name = random.choice(list(self.OPENINGS.keys()))
        # A random variation for that opening is selected
        self.variation = random.choice(self.OPENINGS[opening_name])
        # You can print this to see which opening was chosen for the game
        print(f"Chosen Opening: {opening_name}")

        self.tt = {}

        # Piece values in centipawns (Pawn=100, Knight=320, Bishop=330, Rook=500, Queen=900).
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 0
        }
        # Piece-Square Tables for white (indexed by square 0=a1, ..., 63=h8).
        # Values in centipawns giving positional bonus/penalty for each piece type.
        self.pawn_table = [
            0, 0, 0, 0, 0, 0, 0, 0,
            5, 10, 10,-20,-20, 10, 10, 5,
            5, -5,-10,  0,  0,-10, -5, 5,
            0,  0,  0, 20, 20,  0,  0, 0,
            5,  5, 10, 25, 25, 10,  5, 5,
            10, 10, 20, 30, 30, 20, 10, 10,
            50, 50, 50, 50, 50, 50, 50, 50,
            0,  0,  0,  0,  0,  0,  0,  0
        ]
        self.knight_table = [
            -50,-40,-30,-30,-30,-30,-40,-50,
            -40,-20,  0,  5,  5,  0,-20,-40,
            -30,  5, 10, 15, 15, 10,  5,-30,
            -30,  0, 15, 20, 20, 15,  0,-30,
            -30,  5, 15, 20, 20, 15,  5,-30,
            -30,  0, 10, 15, 15, 10,  0,-30,
            -40,-20,  0,  0,  0,  0,-20,-40,
            -50,-40,-30,-30,-30,-30,-40,-50
        ]
        self.bishop_table = [
            -20,-10,-10,-10,-10,-10,-10,-20,
            -10,  5,  0,  0,  0,  0,  5,-10,
            -10, 10, 10, 10, 10, 10, 10,-10,
            -10,  0, 10, 10, 10, 10,  0,-10,
            -10,  5,  5, 10, 10,  5,  5,-10,
            -10,  0,  5, 10, 10,  5,  0,-10,
            -10,  0,  0,  0,  0,  0,  0,-10,
            -20,-10,-10,-10,-10,-10,-10,-20
        ]
        self.rook_table = [
             0,  0,  0,  5,  5,  0,  0,  0,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
             5, 10, 10, 10, 10, 10, 10,  5,
             0,  0,  0,  0,  0,  0,  0,  0
        ]
        self.queen_table = [
            -20,-10,-10, -5, -5,-10,-10,-20,
            -10,  0,  5,  0,  0,  0,  0,-10,
            -10,  5,  5,  5,  5,  5,  0,-10,
             0,   0,  5,  5,  5,  5,  0, -5,
            -5,   0,  5,  5,  5,  5,  0, -5,
            -10,  0,  5,  5,  5,  5,  0,-10,
            -10,  0,  0,  0,  0,  0,  0,-10,
            -20,-10,-10, -5, -5,-10,-10,-20
        ]
        self.king_table_mid = [
            20, 30, 10,  0,  0, 10, 30, 20,
            20, 20,  0,  0,  0,  0, 20, 20,
           -10,-20,-20,-20,-20,-20,-20,-10,
           -20,-30,-30,-40,-40,-30,-30,-20,
           -30,-40,-40,-50,-50,-40,-40,-30,
           -30,-40,-40,-50,-50,-40,-40,-30,
           -30,-40,-40,-50,-50,-40,-40,-30,
           -30,-40,-40,-50,-50,-40,-40,-30
        ]
        self.king_table_end = [
           -50,-30,-30,-30,-30,-30,-30,-50,
           -30,-30,  0,  0,  0,  0,-30,-30,
           -30,-10, 20, 30, 30, 20,-10,-30,
           -30,-10, 30, 40, 40, 30,-10,-30,
           -30,-10, 30, 40, 40, 30,-10,-30,
           -30,-10, 20, 30, 30, 20,-10,-30,
           -30,-20,-10,  0,  0,-10,-20,-30,
           -50,-40,-30,-20,-20,-30,-40,-50
        ]

    def is_endgame(self, board: chess.Board) -> bool:
        """Determine if the position is an endgame (used to switch king PST)."""
        if board.is_checkmate() or board.is_stalemate():
            return True
        if not board.pieces(chess.QUEEN, chess.WHITE) and not board.pieces(chess.QUEEN, chess.BLACK):
            return True
        white_material = sum(len(board.pieces(pt, chess.WHITE)) * self.piece_values[pt] for pt in [chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.QUEEN])
        black_material = sum(len(board.pieces(pt, chess.BLACK)) * self.piece_values[pt] for pt in [chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.QUEEN])
        if white_material <= 500 and black_material <= 500:
            return True
        return False

    def evaluate(self, board: chess.Board) -> int:
        """
        Evaluate the board position and return a score (centipawns).
        Positive scores favor White, negative favor Black.
        """
        if board.is_checkmate():
            return -100000 if board.turn == chess.WHITE else 100000
        if board.is_stalemate() or board.is_insufficient_material():
            return 0

        king_table = self.king_table_end if self.is_endgame(board) else self.king_table_mid
        score = 0
        for square, piece in board.piece_map().items():
            piece_value = self.piece_values[piece.piece_type]
            positional_value = 0
            
            # Select the correct piece-square table
            if piece.piece_type == chess.PAWN:
                table = self.pawn_table
            elif piece.piece_type == chess.KNIGHT:
                table = self.knight_table
            elif piece.piece_type == chess.BISHOP:
                table = self.bishop_table
            elif piece.piece_type == chess.ROOK:
                table = self.rook_table
            elif piece.piece_type == chess.QUEEN:
                table = self.queen_table
            elif piece.piece_type == chess.KING:
                table = king_table

            # Get positional value from the table
            if piece.color == chess.WHITE:
                positional_value = table[square]
                score += piece_value + positional_value
            else:
                # Mirror the square for Black's perspective
                positional_value = table[chess.square_mirror(square)]
                score -= (piece_value + positional_value)
                
        # Add heuristic bonuses
        score += self.rook_file_bonus(board)
        score += self.bishop_diagonal_bonus(board)
        score += self.knight_outpost_bonus(board)

        return score

    def select_move(self, board: chess.Board) -> chess.Move:
        """
        Choose the best move for the current player using minimax search.
        Plays from the opening book if applicable.
        """
        # Play from opening book if moves are available
        ply = len(board.move_stack)
        if self.variation and ply < len(self.variation):
            san = self.variation[ply]
            try:
                move = board.parse_san(san)
                if move in board.legal_moves:
                    return move
            except ValueError:
                # Fall through to search if the opening move is illegal
                pass

        if board.is_game_over():
            return None

        best_move = None
        alpha = -float('inf')
        beta = float('inf')
        maximizing_player = (board.turn == chess.WHITE)
        
        # Simple move ordering: captures and checks first
        def move_priority(move: chess.Move) -> int:
            score = 0
            if board.is_capture(move):
                captured = board.piece_at(move.to_square)
                if captured: score += 10 * self.piece_values[captured.piece_type]
            if board.gives_check(move):
                score += 5
            return score

        moves = sorted(list(board.legal_moves), key=move_priority, reverse=True)

        if maximizing_player:
            max_eval = -float('inf')
            for move in moves:
                board.push(move)
                eval_score = self._alphabeta(board, self.depth - 1, alpha, beta, False)
                board.pop()
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
        else: # Minimizing player
            min_eval = float('inf')
            for move in moves:
                board.push(move)
                eval_score = self._alphabeta(board, self.depth - 1, alpha, beta, True)
                board.pop()
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
        
        return best_move if best_move is not None else random.choice(list(board.legal_moves))

    def bishop_diagonal_bonus(self, board: chess.Board) -> int:
        bonus = 0
        for sq in board.pieces(chess.BISHOP, chess.WHITE):
            if not any(board.piece_type_at(s) == chess.PAWN for s in board.attacks(sq)):
                bonus += 20
        for sq in board.pieces(chess.BISHOP, chess.BLACK):
            if not any(board.piece_type_at(s) == chess.PAWN for s in board.attacks(sq)):
                bonus -= 20
        return bonus

    def knight_outpost_bonus(self, board: chess.Board) -> int:
        bonus = 0
        for color in [chess.WHITE, chess.BLACK]:
            for sq in board.pieces(chess.KNIGHT, color):
                rank = chess.square_rank(sq)
                # Check if knight is in opponent's half
                if (color == chess.WHITE and rank < 3) or (color == chess.BLACK and rank > 4):
                    continue
                # Check if protected by a friendly pawn and not attackable by an enemy pawn
                if any(board.piece_type_at(s) == chess.PAWN for s in board.attackers(color, sq)) and \
                   not any(board.piece_type_at(s) == chess.PAWN for s in board.attackers(not color, sq)):
                    bonus += 25 if color == chess.WHITE else -25
        return bonus

    def rook_file_bonus(self, board: chess.Board) -> int:
        bonus = 0
        for color in [chess.WHITE, chess.BLACK]:
            for sq in board.pieces(chess.ROOK, color):
                file_index = chess.square_file(sq)
                file_pawns = [s for s in chess.SQUARES_180 if chess.square_file(s) == file_index and board.piece_type_at(s) == chess.PAWN]
                
                if not file_pawns: # Open file
                    bonus += 30 if color == chess.WHITE else -30
                elif not any(board.piece_at(p).color == color for p in file_pawns): # Half-open file
                    bonus += 15 if color == chess.WHITE else -15
        return bonus

    def _alphabeta(self, board: chess.Board, depth: int, alpha: int, beta: int, maximizing: bool) -> int:
        """
        Alpha-beta with transposition table and move ordering optimizations.
        """
        key = polyglot.zobrist_hash(board)
        orig_alpha, orig_beta = alpha, beta
        best_local = None

        # TT lookup
        if key in self.tt:
            entry = self.tt[key]
            if entry.depth >= depth:
                if entry.flag == EXACT:
                    return entry.score
                elif entry.flag == LOWERBOUND:
                    alpha = max(alpha, entry.score)
                elif entry.flag == UPPERBOUND:
                    beta = min(beta, entry.score)
                if alpha >= beta:
                    return entry.score

        # Leaf node evaluation
        if depth == 0 or board.is_game_over():
            score = self.evaluate(board)
        else:
            # Generate and order moves: TT best move first, then captures/checks
            legal = list(board.legal_moves)
            # Move ordering by TT data
            if key in self.tt:
                best_move = self.tt[key].best_move
                if best_move in legal:
                    legal.remove(best_move)
                    legal.insert(0, best_move)
            # Simple heuristics for ordering
            legal.sort(key=lambda m: (board.is_capture(m), board.gives_check(m)), reverse=True)

            score = -float('inf') if maximizing else float('inf')
            best_local = None
            for move in legal:
                board.push(move)
                val = self._alphabeta(board, depth-1, alpha, beta, not maximizing)
                board.pop()

                if maximizing:
                    if val > score:
                        score, best_local = val, move
                    alpha = max(alpha, val)
                else:
                    if val < score:
                        score, best_local = val, move
                    beta = min(beta, val)

                if alpha >= beta:
                    break

        # TT store
        flag = EXACT
        if score <= orig_alpha:
            flag = UPPERBOUND
        elif score >= orig_beta:
            flag = LOWERBOUND
        self.tt[key] = tt_entry(depth, flag, score, best_local)

        return score