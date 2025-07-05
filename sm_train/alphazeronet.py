import chess
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Neural Network Definition for AlphaZero ---
class AlphaZeroNet(nn.Module):
    def __init__(self):
        super(AlphaZeroNet, self).__init__()
        # Small ConvNet: 13 input channels (12 piece planes + side-to-move)
        self.conv1 = nn.Conv2d(13, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        # Policy head: output 73 feature maps (8x8x73=4672 moves)
        self.policy_conv = nn.Conv2d(64, 73, kernel_size=1)
        # Value head: conv to 1 map, then fully connected
        self.value_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.value_fc1 = nn.Linear(8*8*1, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # Policy head
        p = self.policy_conv(x)
        p = p.view(p.size(0), -1)
        # Value head
        v = F.relu(self.value_conv(x))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))
        return p, v

# --- Move Encoding for 8x8x73 planes ---
_codes = {}
_index_to_code = {}
i = 0
dirs = ["N","NE","E","SE","S","SW","W","NW"]
for n in range(1,8):
    for d in dirs:
        _codes[(n,d)] = i; _index_to_code[i] = (n,d); i += 1
knight_moves = []
for two in ["N","S"]:
    for one in ["E","W"]:
        knight_moves.append(("knight",two,one))
for two in ["E","W"]:
    for one in ["N","S"]:
        knight_moves.append(("knight",two,one))
for mv in knight_moves:
    _codes[mv] = i; _index_to_code[i] = mv; i += 1
promo_dirs = ["N","NE","NW"]
promo_pcs = ["knight","bishop","rook"]
for d in promo_dirs:
    for p in promo_pcs:
        _codes[("underpromo",d,p)] = i; _index_to_code[i] = ("underpromo",d,p); i += 1


def move_to_index(move: chess.Move, board: chess.Board) -> int:
    piece = board.piece_at(move.from_square)
    if piece is None:
        raise ValueError("No piece at from_square.")
    ff = chess.square_file(move.from_square)
    rf = chess.square_rank(move.from_square)
    tf = chess.square_file(move.to_square)
    rt = chess.square_rank(move.to_square)
    if piece.color == chess.BLACK:
        ms = chess.square_mirror(move.from_square)
        ts = chess.square_mirror(move.to_square)
        ff = chess.square_file(ms); rf = chess.square_rank(ms)
        tf = chess.square_file(ts); rt = chess.square_rank(ts)
    dx, dy = tf - ff, rt - rf
    # Knight
    if piece.piece_type == chess.KNIGHT:
        if abs(dx)==1 and abs(dy)==2:
            td = "N" if dy>0 else "S"; od = "E" if dx>0 else "W"
        else:
            td = "E" if dx>0 else "W"; od = "N" if dy>0 else "S"
        code = ("knight",td,od)
    # Underpromo
    elif piece.piece_type==chess.PAWN and move.promotion and move.promotion!=chess.QUEEN:
        if dx==0: dstr="N"
        elif dx>0: dstr="NE"
        else: dstr="NW"
        pstr = {chess.KNIGHT:"knight",chess.BISHOP:"bishop",chess.ROOK:"rook"}[move.promotion]
        code = ("underpromo",dstr,pstr)
    else:
        if dx==0 and dy>0: code=(1,"N")
        elif dx==0 and dy<0: code=(1,"S")
        elif dy==0 and dx>0: code=(1,"E")
        elif dy==0 and dx<0: code=(1,"W")
        elif dx>0 and dy>0: code=(abs(dx),"NE")
        elif dx<0 and dy>0: code=(abs(dx),"NW")
        elif dx>0 and dy<0: code=(abs(dx),"SE")
        elif dx<0 and dy<0: code=(abs(dx),"SW")
        else:
            raise ValueError("Unexpected move.")
    base = _codes[code]
    return base + 64*chess.square_rank(move.from_square) + chess.square_file(move.from_square)


def board_to_tensor(board: chess.Board) -> torch.Tensor:
    x = torch.zeros((13,8,8),dtype=torch.float32)
    p2p={chess.PAWN:0,chess.KNIGHT:1,chess.BISHOP:2,chess.ROOK:3,chess.QUEEN:4,chess.KING:5}
    for sq,p in board.piece_map().items():
        idx = p2p[p.piece_type] + (6 if p.color==chess.BLACK else 0)
        x[idx, chess.square_rank(sq), chess.square_file(sq)] = 1.0
    x[12,:,:] = 1.0 if board.turn==chess.WHITE else 0.0
    return x.unsqueeze(0)

class Node:
    def __init__(self, board, parent=None, move=None, prior=0.0):
        self.board=board; self.parent=parent; self.move=move; self.P=prior
        self.N=0; self.W=0.0; self.Q=0.0; self.children={}
    def is_expanded(self): return bool(self.children)
    def expand(self,probs):
        for mv in self.board.legal_moves:
            idx = move_to_index(mv,self.board)
            pr = probs[idx].item()
            if pr>0:
                nb=self.board.copy(); nb.push(mv)
                self.children[mv]=Node(nb,self,mv,pr)
    def select_child(self,c_puct=1.5):
        total=math.sqrt(self.N+1e-8)
        best_score=-1e9; best=None
        for mv,ch in self.children.items():
            u = c_puct*ch.P*total/(1+ch.N)
            sc = ch.Q+u
            if sc>best_score: best_score, best = sc, mv
        return best, self.children[best]
    def backpropagate(self,value):
        node, v = self, value
        while node:
            node.N+=1; node.W+=v; node.Q=node.W/node.N
            v=-v; node=node.parent

class AlphaZeroStrategy:
    def __init__(self, model_path=None, simulations=200):
        self.model=AlphaZeroNet()
        if model_path:
            self.model.load_state_dict(torch.load(model_path,map_location='cpu'))
        self.model.eval()
        self.simulations=simulations
        self.last_root_policy=None
    
    def select_move(self,board):
        mv,_=self._run_mcts(board)
        return mv

    def select_move_with_policy(self,board):
        return self._run_mcts(board)

    def _run_mcts(self,board):
        root=Node(board.copy())
        for _ in range(self.simulations):
            node=root
            # 1) Selection
            while node.is_expanded(): _,node=node.select_child()
            # 2) Evaluation & Expansion
            if not node.board.is_game_over():
                x=board_to_tensor(node.board)
                with torch.no_grad(): logits,val=self.model(x)
                probs=torch.softmax(logits,dim=1)[0]
                mask=torch.zeros_like(probs)
                for mv in node.board.legal_moves:
                    mask[move_to_index(mv,node.board)] = 1
                probs=probs*mask
                if probs.sum()>0: probs=probs/probs.sum()
                node.expand(probs)
                value=val.item()
            else:
                value = -1.0 if node.board.is_checkmate() else 0.0
            # 3) Backpropagation
            node.backpropagate(value)
        # Collect root policy
        policy=torch.zeros(4672)
        for mv,ch in root.children.items():
            idx=move_to_index(mv,root.board)
            policy[idx]=ch.N
        if policy.sum()>0: policy/=policy.sum()
        self.last_root_policy=policy
        # Best move
        best_move=max(root.children.items(),key=lambda itm: itm[1].N)[0]
        return best_move, policy
