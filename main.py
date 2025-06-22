#!/usr/bin/env python
# Minimal random-move Lichess bot – Python 3.9+, berserk 0.14, python-chess 1.999

import os, random, threading, berserk, chess

TOKEN   = os.environ["LICHESS_TOKEN"]          # your API token
session = berserk.TokenSession(TOKEN)
client  = berserk.Client(session=session)
BOT_ID  = client.account.get()["id"].lower()    # e.g. "sammierandombot"

def side_has_bot(side: dict) -> bool:
    """Return True if this side object represents our bot."""
    if not isinstance(side, dict):
        return False
    return (
        str(side.get("id", "")).lower()   == BOT_ID or
        str(side.get("name", "")).lower() == BOT_ID or
        str(side.get("user", {}).get("id", "")).lower()   == BOT_ID or
        str(side.get("user", {}).get("name", "")).lower() == BOT_ID
    )

# ---------------------------------------------------------------------------
def play_game(game_id: str):
    """One thread per game – rebuild board from moves each time."""
    def board_from_moves(moves: str) -> chess.Board:
        b = chess.Board()                 # always start from normal position
        for mv in moves.split():
            b.push_uci(mv)
        return b

    stream     = client.bots.stream_game_state(game_id)
    game_full  = next(stream)             # first packet
    white_side = game_full.get("white", {})
    black_side = game_full.get("black", {})
    my_color   = "white" if side_has_bot(white_side) else "black"

    moves = game_full["state"]["moves"]
    board = board_from_moves(moves)

    # Play immediately if it’s already our turn
    if board.turn == (my_color == "white"):
        legal_moves = list(board.legal_moves)          # ▼
        if not legal_moves:                            # ▼
            return                                     # ▼
        move = random.choice(legal_moves)              # ▼
        client.bots.make_move(game_id, move.uci())
        moves += (" " if moves else "") + move.uci()

    for pkt in stream:
        if pkt["type"] != "gameState" or pkt["moves"] == moves:
            continue
        moves = pkt["moves"]
        board = board_from_moves(moves)
        if board.turn == (my_color == "white"):
            legal_moves = list(board.legal_moves)      # ▼
            if not legal_moves:                        # ▼
                return                                 # ▼
            move = random.choice(legal_moves)          # ▼
            client.bots.make_move(game_id, move.uci())


# ---------------------------------------------------------------------------
print("✅ Bot online – waiting for casual-standard challenges …", flush=True)

def is_from_me(challenge: dict) -> bool:
    challenger = challenge.get("challenger") or {}
    return str(challenger.get("id", "")).lower() == BOT_ID

for evt in client.bots.stream_incoming_events():
    if evt["type"] == "challenge":
        ch = evt["challenge"]

        if is_from_me(ch):
            continue                    # skip self-issued seeks

        if ch["variant"]["key"] == "standard" and not ch["rated"]:
            client.challenges.accept(ch["id"])
        else:
            client.challenges.decline(ch["id"])

    elif evt["type"] == "gameStart":
        threading.Thread(
            target=play_game,
            args=(evt["game"]["id"],),
            daemon=True
        ).start()
