import chess

class OpenAIStrategy:
    """
    A MoveStrategy implementation that selects moves using 
    minimax search with alpha-beta pruning and heuristic evaluation.
    """
    def __init__(self, depth: int = 3):
        self.depth = depth  # search depth in plies (half-moves)
        # Piece values in centipawns (Pawn=100, Knight=320, Bishop=330, Rook=500, Queen=900).
        # King is not included in material evaluation (handled via checkmate detection).
        self.piece_values = {
            chess.PAWN: 100, 
            chess.KNIGHT: 320, 
            chess.BISHOP: 330, 
            chess.ROOK: 500, 
            chess.QUEEN: 900,
            chess.KING: 0    # King's value is set to 0 for evaluation (checkmate handled separately)
        }
        # Piece-Square Tables for white (indexed by square 0= a1, ..., 63 = h8).
        # Values in centipawns giving positional bonus/penalty for each piece type.
        self.pawn_table = [
            # Rank 1 to 8 (from white's perspective)
            0, 0, 0, 0, 0, 0, 0, 0,           # a1 ... h1
            5, 10, 10,-20,-20, 10, 10, 5,     # a2 ... h2
            5, -5,-10,  0,  0,-10, -5, 5,     # a3 ... h3
            0,  0,  0, 20, 20,  0,  0, 0,     # a4 ... h4
            5,  5, 10, 25, 25, 10,  5, 5,     # a5 ... h5
            10, 10, 20, 30, 30, 20, 10, 10,   # a6 ... h6
            50, 50, 50, 50, 50, 50, 50, 50,   # a7 ... h7
            0,  0,  0,  0,  0,  0,  0,  0      # a8 ... h8
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
        # King tables for middle-game and endgame.
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
        # A simple heuristic: consider it endgame if no queens are on the board,
        # or if total non-pawn material is very low.
        if board.is_checkmate() or board.is_stalemate():
            return True  # game over is treated as endgame scenario for evaluation
        # If no queens for either side:
        if not board.pieces(chess.QUEEN, chess.WHITE) and not board.pieces(chess.QUEEN, chess.BLACK):
            return True
        # Calculate total material (excluding pawns and kings) for each side
        white_material = 0
        black_material = 0
        for piece_type in [chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.QUEEN]:
            white_material += len(board.pieces(piece_type, chess.WHITE)) * self.piece_values[piece_type]
            black_material += len(board.pieces(piece_type, chess.BLACK)) * self.piece_values[piece_type]
        # If both sides have very low material (e.g., <= one minor each), consider endgame
        if white_material <= 500 and black_material <= 500:
            # 500 centipawns ~ value of a rook or minor piece
            return True
        return False

    def evaluate(self, board: chess.Board) -> int:
        """
        Evaluate the board position and return a score (centipawns).
        Positive scores favor White, negative favor Black.
        """
        # If game is over, return a high/low value for checkmate or 0 for draw
        if board.is_checkmate():
            # If it's checkmate, the side to move has no moves and is in check.
            # That means the *previous* move delivered mate.
            return (-100000 if board.turn == chess.WHITE else 100000)
        if board.is_stalemate() or board.is_insufficient_material():
            return 0  # draw

        # Select appropriate king table depending on phase
        if self.is_endgame(board):
            king_table = self.king_table_end
        else:
            king_table = self.king_table_mid

        # Sum up material and positional scores
        white_score = 0
        black_score = 0
        for square, piece in board.piece_map().items():
            value = self.piece_values[piece.piece_type]
            # Add piece material value
            if piece.color == chess.WHITE:
                white_score += value
            else:
                black_score += value
            # Add piece-square table value (positional bonus)
            if piece.piece_type == chess.PAWN:
                # Pawns use pawn_table
                if piece.color == chess.WHITE:
                    white_score += self.pawn_table[square]
                else:
                    # mirror the square for black pawn
                    black_score += self.pawn_table[chess.square_mirror(square)]
            elif piece.piece_type == chess.KNIGHT:
                if piece.color == chess.WHITE:
                    white_score += self.knight_table[square]
                else:
                    black_score += self.knight_table[chess.square_mirror(square)]
            elif piece.piece_type == chess.BISHOP:
                if piece.color == chess.WHITE:
                    white_score += self.bishop_table[square]
                else:
                    black_score += self.bishop_table[chess.square_mirror(square)]
            elif piece.piece_type == chess.ROOK:
                if piece.color == chess.WHITE:
                    white_score += self.rook_table[square]
                else:
                    black_score += self.rook_table[chess.square_mirror(square)]
            elif piece.piece_type == chess.QUEEN:
                if piece.color == chess.WHITE:
                    white_score += self.queen_table[square]
                else:
                    black_score += self.queen_table[chess.square_mirror(square)]
            elif piece.piece_type == chess.KING:
                # Use king positional table
                if piece.color == chess.WHITE:
                    white_score += king_table[square]
                else:
                    black_score += king_table[chess.square_mirror(square)]
        # Overall evaluation is material+positional score difference
        return white_score - black_score

    def select_move(self, board: chess.Board) -> chess.Move:
        """
        Choose the best move for the current player (side to move) using minimax search.
        Returns a chess.Move object.
        """
        # If no moves available (game over), return None
        if board.is_game_over():
            return None

        best_move = None
        # Initialize alpha-beta bounds
        alpha = -1000000
        beta = 1000000

        # Determine if we are maximizing or minimizing
        maximizing_player = (board.turn == chess.WHITE)

        if maximizing_player:
            max_eval = -1000000
        else:
            min_eval = 1000000

        # Order moves: prioritize captures and checks for better pruning
        def move_priority(move: chess.Move) -> int:
            score = 0
            # Encourage moves that capture valuable opponent pieces
            if board.is_capture(move):
                captured_piece = board.piece_at(move.to_square)
                if captured_piece:
                    score += 10 * self.piece_values[captured_piece.piece_type]
            # Encourage moves that put opponent in check
            if board.gives_check(move):
                score += 5
            # (We give higher score to moves we want to consider first)
            return score

        moves = list(board.legal_moves)
        moves.sort(key=move_priority, reverse=True)

        # Search each move
        for move in moves:
            board.push(move)
            score = self._alphabeta(board, self.depth - 1, alpha, beta, not maximizing_player)
            board.pop()

            if maximizing_player:
                if score > max_eval:
                    max_eval = score
                    best_move = move
                alpha = max(alpha, max_eval)
            else:
                if score < min_eval:
                    min_eval = score
                    best_move = move
                beta = min(beta, min_eval)

            # Alpha-beta cutoff
            if alpha >= beta:
                break

        return best_move

    def _alphabeta(self, board: chess.Board, depth: int, alpha: int, beta: int, maximizing: bool) -> int:
        """
        Recursively apply minimax with alpha-beta pruning and return the evaluation score.
        """
        # Evaluate at leaf node or terminal position
        if depth == 0 or board.is_game_over():
            return self.evaluate(board)

        if maximizing:
            max_eval = -1000000
            for move in board.legal_moves:
                board.push(move)
                score = self._alphabeta(board, depth - 1, alpha, beta, False)
                board.pop()
                if score > max_eval:
                    max_eval = score
                if max_eval > alpha:
                    alpha = max_eval
                if beta <= alpha:
                    break  # beta cut-off
            return max_eval
        else:
            min_eval = 1000000
            for move in board.legal_moves:
                board.push(move)
                score = self._alphabeta(board, depth - 1, alpha, beta, True)
                board.pop()
                if score < min_eval:
                    min_eval = score
                if min_eval < beta:
                    beta = min_eval
                if beta <= alpha:
                    break  # alpha cut-off
            return min_eval
