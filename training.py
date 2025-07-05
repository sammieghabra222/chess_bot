import time
import torch
import torch.nn.functional as F
import chess
from strategies.alphazeronet import AlphaZeroStrategy, board_to_tensor

# === Configuration ===
num_iterations       = 100   # training rounds
games_per_iteration  = 10    # self‐play games per round
simulations          = 200   # MCTS sims per move
move_limit           = 200   # max moves (ply) per game

# === Initialize ===
engine    = AlphaZeroStrategy(simulations=simulations)
optimizer = torch.optim.Adam(engine.model.parameters(), lr=0.001)

for iteration in range(1, num_iterations+1):
    print(f"\n=== Iteration {iteration}/{num_iterations} ===")
    iteration_start = time.time()
    training_data = []

    # 1) Self-play to generate training examples
    for game_num in range(1, games_per_iteration+1):
        print(f"  Game {game_num}/{games_per_iteration} start")
        board = chess.Board()
        game_states, mcts_policies = [], []

        move_count = 0
        while True:
            # Stop if terminal
            if board.is_game_over():
                print(f"    Game ended by terminal at move {move_count}")
                break

            # Stop on 50-move or repetition draw
            if board.can_claim_draw():
                print(f"    Game claimed draw at move {move_count} (repetition/50-move rule)")
                break

            # Stop at hard move limit
            if move_count >= move_limit:
                print(f"    Move limit reached ({move_limit}), declaring draw")
                break

            move_count += 1
            move_start = time.time()

            move, root_policy = engine.select_move_with_policy(board)
            san = board.san(move)
            # entropy for debugging
            probs   = root_policy[root_policy > 0]
            entropy = -(probs * torch.log(probs)).sum().item()
            print(f"    Move {move_count}: {san} "
                  f"(entropy={entropy:.2f}, time={time.time()-move_start:.2f}s)")

            # record state and policy
            game_states.append(board_to_tensor(board))
            mcts_policies.append(root_policy)
            board.push(move)

        # Determine outcome z
        result = board.result(claim_draw=True)
        if result == "1-0":
            outcome =  1.0
        elif result == "0-1":
            outcome = -1.0
        else:
            outcome =  0.0
        print(f"  Game {game_num} result: {result} (z={outcome}) "
              f"in {move_count} moves")

        # Assemble (s_t, π_t, z_t) for this game
        for idx, state in enumerate(game_states):
            player = 1 if (idx % 2 == 0) else -1
            z = outcome * player
            training_data.append((state, mcts_policies[idx], z))

    # 2) Build batches
    states           = torch.cat([s for s, π, z in training_data], dim=0)
    target_policies  = torch.stack([π for s, π, z in training_data], dim=0)
    target_values    = torch.tensor([z for s, π, z in training_data],
                                    dtype=torch.float32).unsqueeze(1)
    print(f"  Prepared batch: {len(training_data)} examples")

    # 3) Forward & Loss
    pred_policies, pred_values = engine.model(states)
    value_loss  = F.mse_loss(pred_values, target_values)
    policy_loss = -torch.mean((target_policies *
                               F.log_softmax(pred_policies, dim=1)
                              ).sum(dim=1))
    loss = value_loss + policy_loss
    print(f"  Loss = value[{value_loss:.4f}] + policy[{policy_loss:.4f}]")

    # 4) Backprop & Optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("  Model updated")

    # 5) Checkpoint
    ckpt = f"model_iter{iteration}.pth"
    torch.save(engine.model.state_dict(), ckpt)
    print(f"  Saved checkpoint: {ckpt}")

    print(f"=== Iteration done in {time.time() - iteration_start:.1f}s, loss={loss:.4f} ===")
