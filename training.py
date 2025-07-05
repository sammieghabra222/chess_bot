import time
import torch
import torch.nn.functional as F
import chess
from strategies.alphazeronet import AlphaZeroStrategy, board_to_tensor

# === Configuration ===
num_iterations = 100        # training rounds
games_per_iteration = 10    # self‐play games per round
simulations = 200           # MCTS sims per move

# === Initialize ===
engine = AlphaZeroStrategy(simulations=simulations)
optimizer = torch.optim.Adam(engine.model.parameters(), lr=0.001)

for iteration in range(1, num_iterations+1):
    print(f"\n=== Iteration {iteration}/{num_iterations} ===")
    iteration_start = time.time()
    training_data = []

    # 1) Self‐play to generate training examples
    for game_num in range(1, games_per_iteration+1):
        board = chess.Board()
        game_states, mcts_policies = [], []

        print(f"  Game {game_num}/{games_per_iteration} start")
        move_count = 0

        while not board.is_game_over():
            move_count += 1
            move_start = time.time()
            move, root_policy = engine.select_move_with_policy(board)

            san = board.san(move)
            # Compute entropy of the root policy as a rough measure of uncertainty
            probs = root_policy[root_policy > 0]
            entropy = -(probs * torch.log(probs)).sum().item()

            print(f"    Move {move_count}: {san} (policy entropy: {entropy:.3f}, time: {time.time()-move_start:.2f}s)")

            game_states.append(board_to_tensor(board))
            mcts_policies.append(root_policy)
            board.push(move)

        # Game over
        result = board.result()
        if result == "1-0":
            outcome =  1.0
        elif result == "0-1":
            outcome = -1.0
        else:
            outcome =  0.0
        print(f"  Game {game_num} result: {result} (outcome={outcome}) in {move_count} moves")

        # Assign z values
        for idx, state in enumerate(game_states):
            player = 1 if idx % 2 == 0 else -1
            z = outcome * player
            training_data.append((state, mcts_policies[idx], z))

    # 2) Build batches
    batch_start = time.time()
    states = torch.cat([s for s, pi, z in training_data], dim=0)
    target_policies = torch.stack([pi for s, pi, z in training_data], dim=0)
    target_values = torch.tensor([z for s, pi, z in training_data], dtype=torch.float32).unsqueeze(1)
    print(f"  Prepared batch: {len(training_data)} examples in {time.time()-batch_start:.2f}s")

    # 3) Forward pass & loss
    forward_start = time.time()
    pred_policies, pred_values = engine.model(states)
    value_loss  = F.mse_loss(pred_values, target_values)
    policy_loss = -torch.mean((target_policies * F.log_softmax(pred_policies, dim=1)).sum(dim=1))
    loss = value_loss + policy_loss
    print(f"  Forward & loss: value_loss={value_loss.item():.4f}, policy_loss={policy_loss.item():.4f} ({time.time()-forward_start:.2f}s)")

    # 4) Backprop & optimization
    opt_start = time.time()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"  Updated model in {time.time()-opt_start:.2f}s")

    # 5) Checkpoint
    ckpt = f"model_iter{iteration}.pth"
    torch.save(engine.model.state_dict(), ckpt)
    print(f"  Saved checkpoint: {ckpt}")

    print(f"=== Iteration {iteration} done in {time.time() - iteration_start:.2f}s, total loss={loss.item():.4f} ===")
