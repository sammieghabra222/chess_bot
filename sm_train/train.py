#!/usr/bin/env python3
import os
import argparse
import torch
import torch.nn.functional as F
import chess

# we import the module itself so we can monkey-patch its board_to_tensor
import alphazeronet
from alphazeronet import AlphaZeroStrategy

def parse_args():
    parser = argparse.ArgumentParser(description="AlphaZero Self-Play Training on SageMaker")
    # SageMaker injected directories
    parser.add_argument('--model-dir',  type=str,
                        default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--output-dir', type=str,
                        default=os.environ.get('SM_OUTPUT_DIR', '/opt/ml/model'))
    # Hyperparameters
    parser.add_argument('--num-iterations',      type=int,   default=100,
                        help="Total training rounds")
    parser.add_argument('--games-per-iteration', type=int,   default=50,
                        help="Self-play games per round")
    parser.add_argument('--simulations',         type=int,   default=200,
                        help="MCTS simulations per move")
    parser.add_argument('--move-limit',          type=int,   default=200,
                        help="Max half-moves per game")
    parser.add_argument('--lr',                  type=float, default=1e-3,
                        help="Learning rate")
    return parser.parse_args()

def main():
    args = parse_args()

    # ─── 0) Device setup ──────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    # ─── 1) Monkey-patch board_to_tensor so that all MCTS inputs go on GPU ──
    _orig_bt = alphazeronet.board_to_tensor
    def _bt_on_device(board):
        return _orig_bt(board).to(device)
    alphazeronet.board_to_tensor = _bt_on_device

    # Now any call inside AlphaZeroStrategy._run_mcts to board_to_tensor
    # will get you a CUDA tensor if a GPU is available.

    # ─── 2) Initialize your engine & optimizer ───────────────────────────────
    engine = AlphaZeroStrategy(
        simulations=args.simulations,
        noise_eps=0.25, noise_alpha=0.03
    )
    engine.model.to(device)                        # <<< weights onto GPU
    optimizer = torch.optim.Adam(engine.model.parameters(), lr=args.lr)

    # ─── 3) Training loop ────────────────────────────────────────────────────
    for iteration in range(1, args.num_iterations+1):
        print(f"\n=== Iteration {iteration}/{args.num_iterations} ===", flush=True)
        training_data = []

        # 3a) Self-play: now both model & inputs are on GPU
        for game_num in range(1, args.games_per_iteration+1):
            print(f"  Self-play game {game_num}/{args.games_per_iteration}", flush=True)
            board = chess.Board()
            states, policies = [], []
            move_count = 0

            while not board.is_game_over() and move_count < args.move_limit:
                move_count += 1
                mv, root_policy = engine.select_move_with_policy(board)
                states.append(root_policy)           # policy is already a CUDA tensor
                # board_to_tensor was patched → returns CUDA tensor
                policies.append(mv)                 # we actually need (state, policy) pairs
                board.push(mv)

            # Convert result into z ∈ {−1,0,1}
            result = board.result(claim_draw=True)
            z = {'1-0':1.0, '0-1':-1.0}.get(result, 0.0)
            # pair up
            for idx, state in enumerate(states):
                sign = 1 if idx % 2 == 0 else -1
                training_data.append((state, policies[idx], z * sign))

        # 3b) Build your batch (all on GPU already)
        states = torch.cat([s for s,_,_ in training_data], dim=0)
        targ_p = torch.stack([π for _,π,_ in training_data], dim=0)
        targ_v = torch.tensor([z for _,_,z in training_data],
                              dtype=torch.float32).unsqueeze(1).to(device)

        # 3c) Forward / backward
        engine.model.train()
        p_logits, v_pred = engine.model(states)
        v_loss = F.mse_loss(v_pred, targ_v)
        p_loss = -torch.mean((targ_p * F.log_softmax(p_logits, dim=1)).sum(dim=1))
        loss = v_loss + p_loss
        print(f"  Training loss = value[{v_loss:.4f}] + policy[{p_loss:.4f}] = {loss:.4f}", flush=True)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        engine.model.eval()

        # 3d) Save a CPU-side checkpoint
        ckpt = os.path.join(args.output_dir, f"model_iter{iteration}.pth")
        cpu_state = {k: v.cpu() for k, v in engine.model.state_dict().items()}
        torch.save(cpu_state, ckpt)
        print(f"  Saved checkpoint → {ckpt}", flush=True)

if __name__ == "__main__":
    main()
