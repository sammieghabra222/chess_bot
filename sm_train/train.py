# train.py
import argparse
import time
import torch
import torch.nn.functional as F
import chess
from strategies.alphazeronet import AlphaZeroStrategy, board_to_tensor

def main():
    # ─── 1) Parse hyperparameters from CLI ───
    parser = argparse.ArgumentParser(description="AlphaZero Self-Play Training")
    parser.add_argument("--num-iterations",       type=int,   default=50,  help="Total training rounds")
    parser.add_argument("--games-per-iteration",  type=int,   default=20,  help="Self-play games per round")
    parser.add_argument("--simulations",          type=int,   default=500, help="MCTS simulations per move")
    parser.add_argument("--move-limit",           type=int,   default=200, help="Max ply before forcing draw")
    parser.add_argument("--lr",                   type=float, default=0.001, help="Learning rate")
    parser.add_argument("--model-path",           type=str,   default=None,  help="Checkpoint to resume from")
    parser.add_argument("--start-iteration",      type=int,   default=1,    help="Iteration to start from")
    args = parser.parse_args()

    # ─── 2) Setup device and engine ───
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    engine    = AlphaZeroStrategy(model_path=args.model_path,
                                  simulations=args.simulations)
    engine.model.to(device)
    optimizer = torch.optim.Adam(engine.model.parameters(), lr=args.lr)

    # ─── 3) Training loop ───
    for iteration in range(args.start_iteration, args.num_iterations + 1):
        print(f"\n=== Iter {iteration}/{args.num_iterations} ===")
        iter_start = time.time()
        training_data = []

        # 3a) Self-play to collect (s, π, z)
        for game_num in range(1, args.games_per_iteration + 1):
            print(f"  Game {game_num}/{args.games_per_iteration}")
            board = chess.Board()
            game_states, mcts_policies = [], []
            move_count = 0

            while True:
                if board.is_game_over():
                    print(f"    Terminal at move {move_count}")
                    break
                if board.can_claim_draw():
                    print(f"    Draw claim at move {move_count}")
                    break
                if move_count >= args.move_limit:
                    print(f"    Move limit {args.move_limit} reached")
                    break

                move_count += 1
                move_start = time.time()
                move, root_policy = engine.select_move_with_policy(board)
                board.push(move)

                # only log every 10 moves
                #if move_count % 10 == 0:
                #    san = board.san(move)
                #    probs   = root_policy[root_policy > 0]
                #    entropy = -(probs * torch.log(probs)).sum().item()
                #    print(f"    Move {move_count}: {san} "
                #          f"(entropy={entropy:.2f}, {time.time()-move_start:.2f}s)")

                game_states.append(board_to_tensor(board))
                mcts_policies.append(root_policy)

            # determine final outcome z
            result  = board.result(claim_draw=True)
            outcome =  1.0 if result == "1-0" else -1.0 if result == "0-1" else 0.0
            print(f"  Result: {result}, z={outcome}")

            # build training triples
            for idx, state in enumerate(game_states):
                player = 1 if (idx % 2 == 0) else -1
                z = outcome * player
                training_data.append((state, mcts_policies[idx], z))

        # 3b) Build training batch
        states          = torch.cat([s for s, π, z in training_data], dim=0).to(device)
        target_policies = torch.stack([π for s, π, z in training_data], dim=0).to(device)
        target_values   = torch.tensor([z for s, π, z in training_data],
                                       dtype=torch.float32).unsqueeze(1).to(device)
        print(f"  Batch size: {len(training_data)}")

        # 3c) Forward pass & loss
        pred_policies, pred_values = engine.model(states)
        value_loss  = F.mse_loss(pred_values, target_values)
        policy_loss = -torch.mean((target_policies *
                                   F.log_softmax(pred_policies, dim=1)
                                  ).sum(dim=1))
        loss = value_loss + policy_loss
        print(f"  Loss = {value_loss:.4f} + {policy_loss:.4f} = {loss:.4f}")

        # 3d) Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("  Model updated")

        # 3e) Checkpoint
        ckpt = f"model_iter{iteration}.pth"
        torch.save(engine.model.state_dict(), ckpt)
        print(f"  Saved {ckpt} in {time.time() - iter_start:.1f}s")

if __name__ == "__main__":
    main()
