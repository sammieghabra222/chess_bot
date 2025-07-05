# train.py
import argparse
import time
import torch
import torch.nn.functional as F
import chess
from torch.utils.data import TensorDataset, DataLoader
from strategies.alphazeronet import AlphaZeroStrategy, board_to_tensor

def main():
    # ─── 1) Parse hyperparameters ───
    parser = argparse.ArgumentParser("AlphaZero Self-Play Training")
    parser.add_argument("--num-iterations",      type=int,   default=50,   help="Total training rounds")
    parser.add_argument("--games-per-iteration",type=int,   default=20,   help="Games per round")
    parser.add_argument("--simulations",         type=int,   default=500,  help="MCTS sims per move")
    parser.add_argument("--move-limit",          type=int,   default=200,  help="Max ply per game")
    parser.add_argument("--batch-size",          type=int,   default=128,  help="Mini-batch size")
    parser.add_argument("--lr",                  type=float, default=0.001,help="Learning rate")
    parser.add_argument("--model-path",          type=str,   default=None, help="Checkpoint to resume from")
    parser.add_argument("--start-iteration",     type=int,   default=1,    help="Iteration to start from")
    args = parser.parse_args()

    # ─── 2) Setup device & engine ───
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    engine = AlphaZeroStrategy(model_path=args.model_path,
                               simulations=args.simulations)
    engine.model.to(device)
    optimizer = torch.optim.Adam(engine.model.parameters(), lr=args.lr)

    # ─── 3) Training loop ───
    for iteration in range(args.start_iteration, args.num_iterations + 1):
        print(f"\n=== Iter {iteration}/{args.num_iterations} ===")
        iter_start = time.time()
        training_data = []

        # 3a) Self-play collect (s, π, z)
        for game_num in range(1, args.games_per_iteration + 1):
            print(f"  Game {game_num}/{args.games_per_iteration}")
            board = chess.Board()
            game_states, mcts_policies = [], []
            move_count = 0

            while True:
                if board.is_game_over(): break
                if board.can_claim_draw(): break
                if move_count >= args.move_limit: break

                move_count += 1
                move, root_policy = engine.select_move_with_policy(board)
                board.push(move)

                game_states.append(board_to_tensor(board))
                mcts_policies.append(root_policy)

            # final outcome
            result  = board.result(claim_draw=True)
            outcome =  1.0 if result=="1-0" else -1.0 if result=="0-1" else 0.0
            print(f"    Result {result}, z={outcome} in {move_count} moves")

            # assemble (s, π, z)
            for idx, state in enumerate(game_states):
                player = 1 if (idx % 2 == 0) else -1
                z = outcome * player
                training_data.append((state, mcts_policies[idx], z))

        # 3b) Build full tensors and DataLoader
        states          = torch.cat([s for s,π,z in training_data], dim=0)
        target_policies = torch.stack([π for s,π,z in training_data], dim=0)
        target_values   = torch.tensor([z for s,π,z in training_data],
                                       dtype=torch.float32).unsqueeze(1)

        dataset = TensorDataset(states, target_policies, target_values)
        loader  = DataLoader(dataset,
                             batch_size=args.batch_size,
                             shuffle=True,
                             drop_last=True)

        print(f"  Collected {len(dataset)} examples → {len(loader)} mini-batches "
              f"of {args.batch_size}")

        # 3c) Iterate mini-batches
        total_loss = 0.0
        for batch_states, batch_policies, batch_values in loader:
            batch_states   = batch_states.to(device)
            batch_policies = batch_policies.to(device)
            batch_values   = batch_values.to(device)

            pred_p, pred_v = engine.model(batch_states)
            value_loss  = F.mse_loss(pred_v, batch_values)
            policy_loss = -torch.mean((batch_policies *
                                       F.log_softmax(pred_p, dim=1)
                                      ).sum(dim=1))
            loss = value_loss + policy_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_states.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"  Iter {iteration} avg loss: {avg_loss:.4f} "
              f"in {time.time()-iter_start:.1f}s")

        # 3d) Checkpoint
        ckpt = f"model_iter{iteration}.pth"
        torch.save(engine.model.state_dict(), ckpt)
        print(f"  Saved checkpoint {ckpt}")

if __name__ == "__main__":
    main()
