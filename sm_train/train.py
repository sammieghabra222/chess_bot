#!/usr/bin/env python3
import os
import argparse
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import chess
from functools import partial

# We import the module itself so we can monkey-patch its board_to_tensor
import alphazeronet
from alphazeronet import AlphaZeroStrategy

def parse_args():
    parser = argparse.ArgumentParser(description="AlphaZero Self-Play Training on SageMaker or Local")
    # SageMaker injected directories or local defaults
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))  # checkpoints

    # Hyperparameters
    parser.add_argument('--num-iterations', type=int, default=100)
    parser.add_argument('--games-per-iteration', type=int, default=50)
    parser.add_argument('--simulations', type=int, default=200)
    parser.add_argument('--move-limit', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num-workers', type=int, default=None,
                        help="Number of parallel workers for self-play. Defaults to CPU count.")
    parser.add_argument('--mcts-batch-size', type=int, default=8,
                        help="Batch size for MCTS neural net calls.")

    return parser.parse_args()


def run_self_play_game(game_num, args, model_state_dict):
    """
    Played inside each worker: runs one full self-play game and returns training tuples.
    """
    pid = os.getpid()
    print(f"  Worker {pid} starting game {game_num+1}/{args.games_per_iteration}")

    # Initialize engine on CPU
    engine = AlphaZeroStrategy(
        simulations=args.simulations,
        batch_size=args.mcts_batch_size
    )
    engine.model.load_state_dict(model_state_dict)
    engine.model.to("cpu")
    engine.device = torch.device("cpu")

    board = chess.Board()
    states, policies = [], []
    move_count = 0

    # Play until terminal or move limit
    while not board.is_game_over(claim_draw=True) and move_count < args.move_limit:
        move_count += 1
        mv, root_policy = engine.select_move_with_policy(board)
        # Save state & policy
        states.append(alphazeronet.board_to_tensor(board))
        policies.append(root_policy)
        board.push(mv)

    # Game result
    result = board.result(claim_draw=True)
    z = {'1-0': 1.0, '0-1': -1.0}.get(result, 0.0)

    # Build data: (state, policy, value)
    game_data = []
    start_ply = board.ply() - len(policies)
    for i, policy in enumerate(policies):
        # Determine perspective: even ply = white, odd = black
        ply = start_ply + i
        value = z if (ply % 2 == 0) else -z
        game_data.append((states[i], policy, value))

    print(f"  Worker {pid} finished game {game_num+1}. Result: {result}")
    return game_data


def main():
    args = parse_args()

    # Use spawn for safety with CUDA
    mp.set_start_method("spawn", force=True)
    num_workers = args.num_workers or mp.cpu_count()
    print(f"Using {num_workers} parallel workers for self-play.")

    # Device for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Main process using device: {device}", flush=True)

    # Initialize main engine on device
    engine = AlphaZeroStrategy(
        simulations=args.simulations,
        batch_size=args.mcts_batch_size
    )
    engine.model.to(device)
    engine.device = device

    optimizer = torch.optim.Adam(engine.model.parameters(), lr=args.lr)

    for iteration in range(1, args.num_iterations + 1):
        print(f"\n=== Iteration {iteration}/{args.num_iterations} ===", flush=True)

        # Share model to workers via state_dict
        cpu_state = {k: v.cpu() for k, v in engine.model.state_dict().items()}

        # 1) Parallel self-play
        print("Starting self-play...")
        with mp.Pool(processes=num_workers) as pool:
            worker = partial(run_self_play_game,
                             args=args,
                             model_state_dict=cpu_state)
            results = pool.map(worker, range(args.games_per_iteration))

        # Flatten
        training_data = [item for game in results for item in game]
        print(f"Self-play done: {len(training_data)} examples.")
        if not training_data:
            continue

        # 2) Prepare batch tensors
        states_t = torch.cat([s for s, p, z in training_data], dim=0).to(device)
        policies_t = torch.stack([p for s, p, z in training_data], dim=0).to(device)
        values_t = torch.tensor([z for s, p, z in training_data], dtype=torch.float32,
                                 device=device).unsqueeze(1)

        # 3) Train on GPU
        print("Starting training...")
        engine.model.train()
        logits, v_pred = engine.model(states_t)
        v_loss = F.mse_loss(v_pred, values_t)
        p_loss = -torch.sum(policies_t * F.log_softmax(logits, dim=1), dim=1).mean()
        loss = v_loss + p_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        engine.model.eval()

        print(f"  Iter {iteration} loss=Value[{v_loss:.4f}] + Policy[{p_loss:.4f}] = {loss:.4f}")

        # 4) Checkpoint
        os.makedirs(args.output_data_dir, exist_ok=True)
        ckpt = os.path.join(args.output_data_dir, f"model_iter_{iteration}.pth")
        torch.save(engine.model.state_dict(), ckpt)
        print(f"  Saved checkpoint -> {ckpt}", flush=True)

if __name__ == "__main__":
    main()
