import chess
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Neural Network Definition for AlphaZero ---
class AlphaZeroNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 13 input channels (12 piece planes + side-to-move)
        self.conv1 = nn.Conv2d(13, 64, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(64)
        # Policy head: 73 planes
        self.policy_conv = nn.Conv2d(64, 73, kernel_size=1)
        # Value head
        self.value_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.value_fc1  = nn.Linear(8*8*1, 64)
        self.value_fc2  = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # Policy
        p = self.policy_conv(x)
        p = p.view(p.size(0), -1)
        # Value
        v = F.relu(self.value_conv(x))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))
        return p, v

# --- Move encoding ---
_codes = {}
_index_to_code = {}
i = 0
dirs = ["N","NE","E","SE","S","SW","W","NW"]
for n in range(1,8):
    for d in dirs:
        _codes[(n,d)] = i; _index_to_code[i] = (n,d); i += 1
# knight moves
for two in ["N","S"]:
    for one in ["E","W"]:
        _codes[("knight",two,one)] = i; _index_to_code[i] = ("knight",two,one); i += 1
for two in ["E","W"]:
    for one in ["N","S"]:
        _codes[("knight",two,one)] = i; _index_to_code[i] = ("knight",two,one); i += 1
# underpromotions
promo_dirs = ["N","NE","NW"]
promo_pcs  = ["knight","bishop","rook"]
for d in promo_dirs:
    for p in promo_pcs:
        _codes[("underpromo",d,p)] = i; _index_to_code[i] = ("underpromo",d,p); i += 1


def move_to_index(move: chess.Move, board: chess.Board) -> int:
    p = board.piece_at(move.from_square)
    if p is None:
        raise ValueError("No piece at from_square.")
    # mirror for black
    from_sq = move.from_square
    to_sq   = move.to_square
    if p.color == chess.BLACK:
        from_sq = chess.square_mirror(from_sq)
        to_sq   = chess.square_mirror(to_sq)
    ff, rf = chess.square_file(from_sq), chess.square_rank(from_sq)
    tf, rt = chess.square_file(to_sq),   chess.square_rank(to_sq)
    dx, dy = tf - ff, rt - rf
    # knight
    if p.piece_type == chess.KNIGHT:
        if abs(dx)==1 and abs(dy)==2:
            d1 = "N" if dy>0 else "S"
            d2 = "E" if dx>0 else "W"
        elif abs(dx)==2 and abs(dy)==1:
            d1 = "E" if dx>0 else "W"
            d2 = "N" if dy>0 else "S"
        code = ("knight", d1, d2)
    # underpromo
    elif p.piece_type==chess.PAWN and move.promotion and move.promotion!=chess.QUEEN:
        dir_ = "N" if dx==0 else ("NE" if dx>0 else "NW")
        promo_map={chess.KNIGHT:"knight", chess.BISHOP:"bishop", chess.ROOK:"rook"}
        code = ("underpromo", dir_, promo_map[move.promotion])
    # sliders & pushes
    else:
        if dx==0 and dy>0:   dir_,dist="N",dy
        elif dx==0 and dy<0: dir_,dist="S",abs(dy)
        elif dy==0 and dx>0: dir_,dist="E",dx
        elif dy==0 and dx<0: dir_,dist="W",abs(dx)
        elif dx>0 and dy>0 and dx==dy:      dir_,dist="NE",dx
        elif dx<0 and dy>0 and abs(dx)==dy: dir_,dist="NW",dy
        elif dx>0 and dy<0 and dx==abs(dy): dir_,dist="SE",dx
        elif dx<0 and dy<0 and dx==dy:      dir_,dist="SW",abs(dx)
        else:
            raise ValueError(f"Unexpected slide dx={dx},dy={dy}")
        code = (dist, dir_)
    plane = _codes[code]
    return plane * 64 + from_sq


def board_to_tensor(board: chess.Board) -> torch.Tensor:
    x = torch.zeros((13,8,8), dtype=torch.float32)
    p2i={chess.PAWN:0, chess.KNIGHT:1, chess.BISHOP:2,
         chess.ROOK:3, chess.QUEEN:4, chess.KING:5}
    for sq,p in board.piece_map().items():
        idx = p2i[p.piece_type] + (6 if p.color==chess.BLACK else 0)
        x[idx, chess.square_rank(sq), chess.square_file(sq)] = 1.0
    x[12,:,:] = 1.0 if board.turn==chess.WHITE else 0.0
    return x.unsqueeze(0)

# --- MCTS Node ---
class Node:
    def __init__(self, board, parent=None, move=None, prior=0.0):
        self.board    = board
        self.parent   = parent
        self.move     = move
        self.P        = prior
        self.N = 0; self.W = 0.0; self.Q=0.0
        self.children = {}
        self.is_terminal = board.is_game_over(claim_draw=True)

    def is_expanded(self):
        return bool(self.children)

    def expand(self, priors: torch.Tensor):
        for mv in self.board.legal_moves:
            idx = move_to_index(mv, self.board)
            prob = priors[idx].item()
            if prob>0:
                nb = self.board.copy(); nb.push(mv)
                child = Node(nb, parent=self, move=mv, prior=prob)
                self.children[mv] = child

    def select_child(self, c_puct=1.5):
        total = math.sqrt(self.N + 1e-8)
        best_score, best_move = -1e9, None
        for mv, ch in self.children.items():
            u = c_puct * ch.P * total / (1 + ch.N)
            score = ch.Q + u
            if score > best_score:
                best_score, best_move = score, mv
        return best_move, self.children[best_move]

    def backpropagate(self, value: float):
        node, v = self, value
        while node is not None:
            node.N += 1
            node.W += v
            node.Q  = node.W / node.N
            v = -v
            node = node.parent

# --- AlphaZero Strategy with batching support ---
class AlphaZeroStrategy:
    def __init__(self, model_path=None, simulations=200,
                 noise_eps=0.25, noise_alpha=0.03, batch_size=8):
        self.model        = AlphaZeroNet()
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()
        self.simulations  = simulations
        self.noise_eps    = noise_eps
        self.noise_alpha  = noise_alpha
        self.batch_size   = batch_size
        # device will be set by user
        self.device       = torch.device('cpu')

    def select_move(self, board):
        mv,_ = self._run_mcts(board)
        return mv

    def select_move_with_policy(self, board):
        return self._run_mcts(board)

    def _run_mcts(self, board: chess.Board):
        root = Node(board.copy())
        # --- INITIAL EXPANSION ---
        x = board_to_tensor(root.board).to(self.device)
        with torch.no_grad():
            logits, _ = self.model(x)
        priors = torch.softmax(logits, dim=1)[0]
        # mask & normalize
        mask = torch.zeros_like(priors)
        for mv in root.board.legal_moves:
            mask[move_to_index(mv, root.board)] = 1
        priors *= mask
        if priors.sum() > 0:
            priors /= priors.sum()
        # add noise
        moves = list(root.board.legal_moves)
        noise = np.random.dirichlet([self.noise_alpha] * len(moves))
        for i, mv in enumerate(moves):
            idx = move_to_index(mv, root.board)
            priors[idx] = priors[idx] * (1 - self.noise_eps) + noise[i] * self.noise_eps
        root.expand(priors)

        # --- SIMULATIONS WITH BATCHING ---
        sims_done = 0
        while sims_done < self.simulations:
            batch_count = min(self.batch_size, self.simulations - sims_done)
            leaf_nodes = []
            # selection
            for _ in range(batch_count):
                node = root
                while node.is_expanded():
                    _, node = node.select_child()
                if node.is_terminal:
                    result = node.board.result(claim_draw=True)
                    value = {'1-0':1.0,'0-1':-1.0}.get(result, 0.0)
                    node.backpropagate(value)
                else:
                    leaf_nodes.append(node)
            if not leaf_nodes:
                sims_done += batch_count
                continue
            # batch evaluate
            batch_states = torch.cat([board_to_tensor(n.board) for n in leaf_nodes], dim=0).to(self.device)
            with torch.no_grad():
                logits_b, values_b = self.model(batch_states)
            priors_b = torch.softmax(logits_b, dim=1)
            # expand & backpropagate
            for i, node in enumerate(leaf_nodes):
                pri = priors_b[i]
                mask = torch.zeros_like(pri)
                for mv in node.board.legal_moves:
                    mask[move_to_index(mv,node.board)] = 1
                pri *= mask
                if pri.sum() > 0:
                    pri /= pri.sum()
                node.expand(pri)
                node.backpropagate(values_b[i].item())
            sims_done += batch_count

        # --- FINAL POLICY & MOVE ---
        policy = torch.zeros(4672, device=self.device)
        for mv, ch in root.children.items():
            idx = move_to_index(mv, root.board)
            policy[idx] = ch.N
        if policy.sum() > 0:
            policy /= policy.sum()
        best_move = max(root.children.items(), key=lambda it: it[1].N)[0]
        return best_move, policy
