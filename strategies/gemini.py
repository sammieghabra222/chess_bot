import chess
import chess.polyglot
import random
import time
from typing import Optional

# --- Part 1: Evaluation Constants and Tables ---

# Piece values
PIECE_VALUES_MG = {
    chess.PAWN: 82, chess.KNIGHT: 337, chess.BISHOP: 365, chess.ROOK: 477, chess.QUEEN: 1025, chess.KING: 0
}
PIECE_VALUES_EG = {
    chess.PAWN: 94, chess.KNIGHT: 281, chess.BISHOP: 372, chess.ROOK: 512, chess.QUEEN: 936, chess.KING: 0
}

# PeSTO's Piece-Square Tables (sourced from the chess programming wiki)
# These are crucial for positional evaluation.
PAWN_PST_MG = [
    0, 0, 0, 0, 0, 0, 0, 0,
    98, 134, 61, 95, 68, 126, 34, -11,
    -6, 7, 26, 31, 65, 56, 25, -20,
    -14, 13, 6, 21, 23, 12, 17, -23,
    -27, -2, -5, 12, 17, 6, 10, -25,
    -26, -4, -4, -10, 3, 3, 33, -12,
    -35, -1, -20, -23, -15, 24, 38, -22,
    0, 0, 0, 0, 0, 0, 0, 0,
]
#... (All other 11 PSTs would be defined here similarly)
# For brevity, only one is shown. The full implementation would include all tables from Chapters 5.

# A simplified set of tables for demonstration if PeSTO's are not fully typed out.
# In a real implementation, all 12 tables from the report would be here.
KNIGHT_PST_MG = [ -167, -89, -34, -49,  61, -97, -15, -107, -73, -41,  72,  36,  23,  62,   7,  -17, -47,  60,  37,  65,  84, 129,  73,   44,  -9,  17,  19,  53,  37,  69,  18,   22, -13,   4,  16,  13,  28,  19,  21,   -8, -23,  -9,  12,  10,  19,  17,  25,  -16, -29, -53, -12,  -3,  -1,  18, -14,  -19, -105, -21, -58, -33, -17, -28, -19,  -23]
BISHOP_PST_MG = [ -29,   4, -82, -37, -25, -42,   7,  -8, -26,  16, -18, -13,  30,  59,  18, -47, -16,  37,  43,  40,  35,  50,  37,  -2,  -4,   5,  19,  50,  37,  37,   7,  -2,  -6,  13,  13,  26,  34,  12,  10,   4,   0,  15,  15,  15,  14,  27,  18,  10,   4,  15,  16,   0,   7,  21,  33,   1, -33,  -3, -14, -21, -13, -12, -39, -21]
ROOK_PST_MG = [ 32,  42,  32,  51, 63,  9,  31,  43, 27,  32,  58,  62, 80, 67,  26,  18, -5,  19,  26,  36, 17, 45,  61,  16, -24, -11,   7,  26, 24, 35,  -8, -20, -36, -26, -12,  -1,  9, -7,   6, -23, -45, -25, -16, -17,  3,  0,  -5, -33, -44, -16, -20,  -9, -1, 11,  -6, -71, -19, -13,   1,  17, 16,  7, -37, -26]
QUEEN_PST_MG = [ -28,   0,  29,  12,  59,  44,  43,  45, -24, -39,  -5,   1, -16,  57,  28,  54, -13, -17,   7,   8,  29,  56,  47,  57, -27, -27, -16, -16,  -1,  17,  -2,   1,  -9, -26,  -9, -10,  -2,  -4,   3,  -3, -14,   2, -11,  -2,  -5,   2,  14,   5, -35,  -8,  11,   2,   8,  15,  -3,   1,  -1, -18,  -9,  10, -15, -25, -31, -50]
KING_PST_MG = [ -65,  23,  16, -15, -56, -34,   2,  13, 29,  -1, -20,  -7,  -8,  -4, -38, -29, -9,  24,   2, -16, -20,   6,  22, -22, -17, -20, -12, -27, -30, -25, -14, -36, -49,  -1, -27, -39, -46, -44, -33, -51, -14, -14, -22, -46, -44, -30, -15, -27,   1,   7,  -8, -64, -43, -16,   9,   8, -15,  36,  12, -54,   8, -28,  24,  14]
PAWN_PST_EG = [0, 0, 0, 0, 0, 0, 0, 0, 178, 173, 158, 134, 147, 132, 165, 187, 94, 100, 85, 67, 56, 53, 82, 84, 32, 24, 13, 5, -2, 4, 17, 17, 4, 2, -2, -7, -17, -4, -18, -31, 2, -2, -12, -21, -34, -40, -45, -57, 6, 2, 0, -5, -7, -13, -24, -41, 0, 0, 0, 0, 0, 0, 0, 0]
KNIGHT_PST_EG = [-58, -38, -13, -28, -31, -27, -63, -99, -25, -8, -25, -2, -9, -25, -24, -52, -24, -20, 10, 9, -1, -9, -19, -41, -17, 3, 22, 22, 22, 11, 8, -18, -18, -6, 16, 25, 16, 17, 4, -18, -23, -3, -1, 15, 10, -3, -20, -22, -42, -20, -10, -5, -2, -20, -23, -44, -29, -51, -23, -15, -22, -18, -50, -64]
BISHOP_PST_EG = [-14, -21, -11, -8, -7, -9, -17, -24, -8, -4, 7, -12, -3, -13, -4, -14, 2, -8, 0, -1, -2, 6, 0, 4, -3, 9, 12, 9, 14, 10, 3, 2, -6, 3, 13, 19, 7, 10, -3, -9, -12, -3, 8, 10, 13, 3, -7, -15, -14, -18, -7, -1, 4, -9, -15, -27, -23, -9, -23, -5, -9, -16, -5, -17]
ROOK_PST_EG = [13, 10, 18, 15, 12, 12, 8, 5, 11, 13, 13, 11, -3, 3, 8, 3, 7, 7, 7, 5, 4, -3, -5, -3, 4, 3, 13, 1, 2, 1, -1, 2, 3, 5, 8, 4, -5, -6, -8, -11, -4, 0, -5, -1, -7, -12, -8, -16, -6, -6, 0, 2, -9, -9, -11, -3, 0, 0, 0, -1, -4, -6, 1, 0]
QUEEN_PST_EG = [-9, 22, 22, 27, 27, 19, 10, 20, -17, 20, 32, 41, 58, 25, 30, 0, -20, 6, 9, 49, 47, 35, 19, 9, 3, 22, 24, 45, 57, 40, 57, 36, -18, 28, 19, 47, 31, 34, 39, 23, -16, -27, 15, 6, 9, 17, 10, 5, -22, -23, -30, -16, -16, -23, -36, -32, -33, -28, -22, -43, -5, -32, -20, -41]
KING_PST_EG = [-74, -35, -18, -18, -11, 15, 4, -17, -12, 17, 14, 17, 17, 38, 23, 11, 10, 17, 23, 15, 20, 45, 44, 13, -8, 22, 24, 27, 26, 33, 26, 3, -18, -4, 21, 24, 27, 23, 9, -11, -19, -3, 11, 21, 23, 16, 7, -9, -27, -11, 4, 13, 15, 4, -5, -17, -53, -34, -21, -11, -28, -14, -24, -43]

PSTS_MG = {
    chess.PAWN: PAWN_PST_MG, chess.KNIGHT: KNIGHT_PST_MG, chess.BISHOP: BISHOP_PST_MG,
    chess.ROOK: ROOK_PST_MG, chess.QUEEN: QUEEN_PST_MG, chess.KING: KING_PST_MG
}
PSTS_EG = {
    chess.PAWN: PAWN_PST_EG, chess.KNIGHT: KNIGHT_PST_EG, chess.BISHOP: BISHOP_PST_EG,
    chess.ROOK: ROOK_PST_EG, chess.QUEEN: QUEEN_PST_EG, chess.KING: KING_PST_EG
}

# Game phase values
PHASE_VALUES = {
    chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 1, chess.ROOK: 2, chess.QUEEN: 4, chess.KING: 0
}
TOTAL_PHASE = PHASE_VALUES[chess.PAWN] * 16 + \
              PHASE_VALUES[chess.KNIGHT] * 4 + \
              PHASE_VALUES[chess.BISHOP] * 4 + \
              PHASE_VALUES[chess.ROOK] * 4 + \
              PHASE_VALUES[chess.QUEEN] * 2
# TOTAL_PHASE = 24

# --- Part 2: The Strategy Class ---

class Gemini25ProStrategy:
    def __init__(self, book_path: str = "baron30.bin", search_depth: int = 4):
        self.book_path = book_path
        self.book_reader = None
        self.fixed_depth = search_depth # Used if time management is off
        self.transposition_table = {}
        self.killer_moves = [[None, None] for _ in range(100)] # Max depth 100
        # Add other initializations like history heuristic table

    def _evaluate(self, board: chess.Board) -> int:
            if board.is_checkmate():
                # Return a very high/low value for checkmate, but avoid infinity
                # to allow for move ordering (e.g., mate in 3 is better than mate in 5)
                # The value is adjusted by ply to find the fastest mate.
                return -99999 + board.ply() if board.turn else 99999 - board.ply()
            if board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves() or board.is_fivefold_repetition():
                return 0

            mg_score = 0
            eg_score = 0
            game_phase = 0

            # Iterate through all piece types
            for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]:
                # White pieces
                for sq in board.pieces(piece_type, chess.WHITE):
                    mg_score += PIECE_VALUES_MG[piece_type] + PSTS_MG[piece_type][sq]
                    eg_score += PIECE_VALUES_EG[piece_type] + PSTS_EG[piece_type][sq]
                    game_phase += PHASE_VALUES[piece_type]
                # Black pieces
                for sq in board.pieces(piece_type, chess.BLACK):
                    # For black, we use square_mirror to get the equivalent PST value
                    mirrored_sq = chess.square_mirror(sq)
                    mg_score -= PIECE_VALUES_MG[piece_type] + PSTS_MG[piece_type][mirrored_sq]
                    eg_score -= PIECE_VALUES_EG[piece_type] + PSTS_EG[piece_type][mirrored_sq]
                    game_phase += PHASE_VALUES[piece_type]
            
            # Tapered evaluation using the game phase
            # Clamp phase to avoid division by zero if all pieces are gone
            phase = min(game_phase, TOTAL_PHASE)
            
            # This formula can cause issues if TOTAL_PHASE is 0
            if TOTAL_PHASE == 0:
                final_score = mg_score # Or handle as a special case
            else:
                final_score = ((mg_score * phase) + (eg_score * (TOTAL_PHASE - phase))) / TOTAL_PHASE

            # Return score from the perspective of the side to move
            return int(final_score) if board.turn == chess.WHITE else -int(final_score)

    def _order_moves(self, board: chess.Board, legal_moves: list) -> list:
        """
        Orders moves based on their value, using a scoring heuristic (MVV-LVA).
        """
        return sorted(legal_moves, key=lambda move: self._get_move_value(board, move), reverse=True)

    def _quiescence(self, board: chess.Board, alpha: float, beta: float) -> int:
        stand_pat = self._evaluate(board)
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat

        # Search captures and non-capturing checks
        moves_to_search = []
        for move in board.legal_moves:
            if board.is_capture(move) or board.gives_check(move):
                moves_to_search.append(move)

        # Order these important moves before searching
        for move in self._order_moves(board, moves_to_search):
            board.push(move)
            score = -self._quiescence(board, -beta, -alpha)
            board.pop()

            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        return alpha

    def _get_move_value(self, board: chess.Board, move: chess.Move) -> int:
        """
        Assigns a score to a move for ordering, prioritizing valuable captures.
        """
        if board.is_capture(move):
            # En-passant capture, where the 'to' square is empty
            if board.is_en_passant(move):
                return PIECE_VALUES_MG[chess.PAWN] + 500  # Give it a high score

            attacker = board.piece_at(move.from_square)
            victim = board.piece_at(move.to_square)
            
            # In case of promotion captures, the attacker is a pawn
            attacker_value = PIECE_VALUES_MG[attacker.piece_type]
            victim_value = PIECE_VALUES_MG[victim.piece_type]
            
            # MVV-LVA: High victim value and low attacker value is best
            return victim_value - attacker_value + 1000
        
        # Placeholder for killer moves/history heuristic in the future
        return 0

    def _negamax(self, board: chess.Board, depth: int, alpha: float, beta: float) -> int:
        zobrist_key = chess.polyglot.zobrist_hash(board)
        if zobrist_key in self.transposition_table and self.transposition_table[zobrist_key]['depth'] >= depth:
            entry = self.transposition_table[zobrist_key]
            if entry['flag'] == 'EXACT':
                return entry['score']
            elif entry['flag'] == 'LOWERBOUND':
                alpha = max(alpha, entry['score'])
            elif entry['flag'] == 'UPPERBOUND':
                beta = min(beta, entry['score'])
            if alpha >= beta:
                return entry['score']

        if depth == 0 or board.is_game_over():
            return self._quiescence(board, alpha, beta)

        best_value = -float('inf')
        ordered_moves = self._order_moves(board, list(board.legal_moves))
        
        for move in ordered_moves:
            board.push(move)
            value = -self._negamax(board, depth - 1, -beta, -alpha)
            board.pop()
            
            if value > best_value:
                best_value = value
            alpha = max(alpha, value)
            if alpha >= beta:
                break # Pruning

        # Store in TT
        flag = 'EXACT'
        if best_value <= alpha: flag = 'UPPERBOUND'
        elif best_value >= beta: flag = 'LOWERBOUND'
        self.transposition_table[zobrist_key] = {'score': best_value, 'depth': depth, 'flag': flag}
        
        return best_value

    def _iterative_deepening_search(self, board: chess.Board, time_limit: float) -> Optional[chess.Move]:
        start_time = time.time()
        best_move = None
        
        # A simple iterative deepening loop
        for depth in range(1, 100): # Max depth of 100
            current_best_move = self._search_at_depth(board, depth)
            if time.time() - start_time > time_limit:
                break
            best_move = current_best_move
        
        return best_move

    def _search_at_depth(self, board: chess.Board, depth: int) -> Optional[chess.Move]:
        best_move = None
        best_value = -float('inf')
        alpha = -float('inf')
        beta = float('inf')

        ordered_moves = self._order_moves(board, list(board.legal_moves))

        for move in ordered_moves:
            board.push(move)
            board_value = -self._negamax(board, depth - 1, -beta, -alpha)
            board.pop()
            if board_value > best_value:
                best_value = board_value
                best_move = move
            alpha = max(alpha, board_value)

        return best_move

    def select_move(self, board: chess.Board) -> Optional[chess.Move]:
        # 1. Initialize opening book reader on first call
        if self.book_reader is None:
            try:
                self.book_reader = chess.polyglot.open_reader(self.book_path)
            except FileNotFoundError:
                print(f"Warning: Opening book '{self.book_path}' not found.")
                self.book_reader = False # Mark as failed to avoid retries

        # 2. Try to find a move in the opening book
        if self.book_reader:
            try:
                return self.book_reader.weighted_choice(board).move
            except IndexError:
                # Position not in book, proceed to search
                pass

        # 3. If not in book, perform an iterative deepening search with a time limit
        self.transposition_table.clear()
        
        # Use iterative deepening to prevent "freezing" and adapt to position complexity.
        # Set a time limit (e.g., 5 seconds). This can be adjusted.
        try:
            return self._iterative_deepening_search(board, time_limit=5.0)
        except Exception as e:
            print(f"An error occurred during search: {e}")
            # As a fallback, return a random legal move if search fails
            return random.choice(list(board.legal_moves))