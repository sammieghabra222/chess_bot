#!/usr/bin/env python
"""
Lichess BOT skeleton with pluggable move-selection strategies.
Python 3.9+, berserk 0.14, python-chess 1.999
"""

import os, random, threading, berserk, chess, traceback
from typing import Optional, Protocol
from strategies import RandomPlayer, MoveStrategy, Gemini25ProStrategy, OpenAIStrategy    # << new


# ──────────────────────── 2. Boilerplate setup ──────────────────────────
# It's good practice to handle the case where the environment variable might be missing.
if "LICHESS_TOKEN" not in os.environ:
    raise ValueError("LICHESS_TOKEN environment variable not set!")

TOKEN   = os.environ["LICHESS_TOKEN"]
session = berserk.TokenSession(TOKEN)
client  = berserk.Client(session=session)
BOT_ID  = client.account.get()["id"].lower()

#STRATEGY: MoveStrategy = Gemini25ProStrategy()
STRATEGY: MoveStrategy = OpenAIStrategy()

def side_has_bot(side: dict) -> bool:
    """Checks if the bot is playing on the given side."""
    sid = lambda k: str(k).lower()
    return sid(side.get("id", "")).lower() == BOT_ID \
        or sid(side.get("name", "")) == BOT_ID \
        or sid(side.get("user", {}).get("id", "")) == BOT_ID \
        or sid(side.get("user", {}).get("name", "")) == BOT_ID


# ───────────────────────── 3. Per-game worker ────────────────────────────
def play_game(game_id: str):
    """
    Handles all logic for a single game.
    This function is run in a separate thread for each game.
    """
    try:
        print(f"[{game_id}] New game thread started.", flush=True)

        def board_from_moves(moves: str) -> chess.Board:
            """Creates a python-chess Board object from a space-separated UCI move string."""
            b = chess.Board()
            if moves: # The move string can be empty
                for mv in moves.split():
                    try:
                        b.push_uci(mv)
                    except ValueError:
                        print(f"[{game_id}] Invalid move '{mv}' in history. Skipping.", flush=True)
            return b

        # This is the first API call, and where the user's error occurred.
        stream = client.bots.stream_game_state(game_id)
        game_full  = next(stream)

        # Determine our color
        if side_has_bot(game_full.get("white", {})):
            my_color = chess.WHITE
            my_color_str = "white"
        else:
            my_color = chess.BLACK
            my_color_str = "black"
        print(f"[{game_id}] Playing as {my_color_str}.", flush=True)

        moves = game_full["state"]["moves"]
        board = board_from_moves(moves)

        # If the game starts and it's already our turn, make a move.
        if not board.is_game_over() and board.turn == my_color:
            move = STRATEGY.select_move(board)
            if move:
                print(f"[{game_id}] Making opening move: {move.uci()}", flush=True)
                client.bots.make_move(game_id, move.uci())
                # No need to update 'moves' here, the stream will send the new state.

        # Main loop for processing game events from the stream.
        for pkt in stream:
            if pkt["type"] == "gameState":
                # Don't re-process the same state if the stream sends duplicates.
                if pkt["moves"] == moves:
                    continue
                
                moves = pkt["moves"]
                board = board_from_moves(moves)

                if board.is_game_over():
                    print(f"[{game_id}] Game over. Result: {board.result()}", flush=True)
                    break # Exit the loop, thread will terminate.

                # It's our turn to move.
                if board.turn == my_color:
                    move = STRATEGY.select_move(board)
                    if move:
                        try:
                            client.bots.make_move(game_id, move.uci())
                            print(f"[{game_id}] Made move: {move.uci()}", flush=True)
                        except berserk.exceptions.ResponseError as e:
                            # This can happen if we try to move in a game that just ended.
                            print(f"[{game_id}] Error making move {move.uci()}: {e}", flush=True)
                    else:
                        # This implies the strategy found no legal moves.
                        print(f"[{game_id}] Strategy returned no moves. Game should be over.", flush=True)
                        break # Exit loop
            
            elif pkt["type"] == "chatLine":
                # Basic chat handling can be added here.
                pass

    # --- Graceful Error Handling ---
    except berserk.exceptions.ResponseError as e:
        if e.status_code == 429:
            print(f"[{game_id}] RATE LIMIT EXCEEDED. Lichess is throttling us.", flush=True)
            print("This usually happens if the bot plays/resigns games too quickly.", flush=True)
            print("The thread for this game will now exit.", flush=True)
        else:
            print(f"[{game_id}] Unrecoverable HTTP Error on game stream: {e}", flush=True)
            
    except Exception as e:
        print(f"[{game_id}] An unexpected error occurred in the game thread:", flush=True)
        traceback.print_exc()
    
    finally:
        print(f"[{game_id}] Thread finished.", flush=True)


# ─────────────────────────── 4. Event loop ──────────────────────────────
def is_from_me(ch: dict) -> bool:
    """Checks if a challenge was sent by our bot."""
    challenger = ch.get("challenger") or {}
    return str(challenger.get("id", "")).lower() == BOT_ID


print("✅ Bot online – waiting for casual-standard challenges …", flush=True)

# A set to keep track of games we are currently playing. This is a good practice
# to prevent starting multiple threads for the same game if Lichess sends duplicate events.
active_game_ids = set()

for evt in client.bots.stream_incoming_events():
    if evt["type"] == "challenge":
        ch = evt["challenge"]
        if is_from_me(ch):
            continue
        if ch["variant"]["key"] == "standard" and not ch["rated"]:
            try:
                client.challenges.accept(ch["id"])
            except berserk.exceptions.ResponseError as e:
                print(f"Failed to accept challenge {ch['id']}: {e}", flush=True)
        else:
            try:
                client.challenges.decline(ch["id"])
            except berserk.exceptions.ResponseError as e:
                # Less critical, but good to log.
                print(f"Failed to decline challenge {ch['id']}: {e}", flush=True)

    elif evt["type"] == "gameStart":
        game_id = evt["game"]["id"]
        if game_id in active_game_ids:
            print(f"[{game_id}] Ignoring duplicate gameStart event.", flush=True)
            continue

        # Add game_id to our active set and define a wrapper to remove it upon completion.
        active_game_ids.add(game_id)
        def thread_wrapper(gid):
            try:
                play_game(gid)
            finally:
                if gid in active_game_ids:
                    active_game_ids.remove(gid)
        
        # Start the game-playing logic in a new thread.
        threading.Thread(target=thread_wrapper,
                         args=(game_id,),
                         daemon=True).start()