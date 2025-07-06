# train.py
import os
import time
import argparse

import torch
import torch.nn.functional as F
import chess

from strategies.alphazeronet import AlphaZeroStrategy, board_to_tensor

def parse_args():
    parser = argparse.ArgumentParser()
    # SageMaker will inject these two
    parser.add_argument('--model-dir',    type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--output-dir',   type=str, default=os.environ.get('SM_OUTPUT_DIR', '/opt/ml/model'))
    # your hyperparameters
    parser.add_argument('--num-iterations',      type=int,   default=100)
    parser.add_argument('--games-per-iteration', type=int,   default=50)
    parser.add_argument('--simulations',         type=int,   default=200)
    parser.add_argument('--move-limit',          type=int,   default=200)
    parser.add_argument('--lr',                  type=float, default=1e-3)
    return parser.parse_args()

def main():
    args = parse_args()

    # 1) Initialize engine & optimizer
    engine    = AlphaZeroStrategy(
                   simulations=args.simulations,
                   noise_eps=0.25, noise_alpha=0.03
                )
    optimizer = torch.optim.Adam(engine.model.parameters(), lr=args.lr)

    # 2) Training loop
    for iteration in range(1, args.num_iterations+1):
        print(f"\n=== Iteration {iteration}/{args.num_iterations} ===")
        training_data = []

        # Self-play
        for game_num in range(1, args.games_per_iteration+1):
            board = chess.Board()
            states, policies = [], []
            move_count = 0

            while True:
                if board.is_game_over():                                 break
                if move_count >= args.move_limit:                        break
                move_count += 1

                move, root_policy = engine.select_move_with_policy(board)
                states.append(board_to_tensor(board))
                policies.append(root_policy)
                board.push(move)

            # outcome z
            res = board.result(claim_draw=True)
            outcome = {'1-0':1.0,'0-1':-1.0}.get(res, 0.0)
            for idx, s in enumerate(states):
                player = 1 if idx % 2 == 0 else -1
                training_data.append((s, policies[idx], outcome * player))

        # Build batch
        states  = torch.cat([s for s,_,_ in training_data], dim=0)
        targ_p  = torch.stack([π for _,π,_ in training_data], dim=0)
        targ_v  = torch.tensor([z for _,_,z in training_data],
                               dtype=torch.float32).unsqueeze(1)

        engine.model.train()
        p_logits, v_pred = engine.model(states)
        value_loss  = F.mse_loss(v_pred, targ_v)
        policy_loss = -torch.mean((targ_p * F.log_softmax(p_logits, dim=1)).sum(dim=1))
        loss = value_loss + policy_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        engine.model.eval()

        # Save checkpoint every iteration
        ckpt_path = os.path.join(args.output_dir, f'model_iter{iteration}.pth')
        torch.save(engine.model.state_dict(), ckpt_path)
        print(f"Saved {ckpt_path}")

if __name__ == '__main__':
    main()
