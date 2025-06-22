#!/usr/bin/env python
"""
Lichess BOT skeleton with pluggable move-selection strategies.
Python 3.9+, berserk 0.14, python-chess 1.999
"""

import os, random, threading, berserk, chess
from typing import Optional, Protocol
from strategies import RandomPlayer, MoveStrategy, Gemini25ProStrategy    # << new


# ──────────────────────── 2. Boilerplate setup ──────────────────────────
TOKEN   = os.environ["LICHESS_TOKEN"]
session = berserk.TokenSession(TOKEN)
client  = berserk.Client(session=session)
BOT_ID  = client.account.get()["id"].lower()

STRATEGY: MoveStrategy = Gemini25ProStrategy()

def side_has_bot(side: dict) -> bool:
    sid = lambda k: str(k).lower()
    return sid(side.get("id", "")).lower() == BOT_ID \
        or sid(side.get("name", "")) == BOT_ID \
        or sid(side.get("user", {}).get("id", "")) == BOT_ID \
        or sid(side.get("user", {}).get("name", "")) == BOT_ID


# ───────────────────────── 3. Per-game worker ────────────────────────────
def play_game(game_id: str):
    def board_from_moves(moves: str) -> chess.Board:
        b = chess.Board()
        for mv in moves.split():
            b.push_uci(mv)
        return b

    stream     = client.bots.stream_game_state(game_id)
    game_full  = next(stream)
    my_color   = "white" if side_has_bot(game_full.get("white", {})) else "black"

    moves = game_full["state"]["moves"]
    board = board_from_moves(moves)

    # play at game start if it's already our move
    if board.turn == (my_color == "white"):
        move = STRATEGY.select_move(board)
        if move:
            client.bots.make_move(game_id, move.uci())
            moves += (" " if moves else "") + move.uci()

    # main update loop
    for pkt in stream:
        if pkt["type"] != "gameState" or pkt["moves"] == moves:
            continue
        moves = pkt["moves"]
        board = board_from_moves(moves)

        if board.turn == (my_color == "white"):
            move = STRATEGY.select_move(board)
            if move:
                client.bots.make_move(game_id, move.uci())
            else:               # no legal moves ⇒ game over
                return


# ─────────────────────────── 4. Event loop ──────────────────────────────
def is_from_me(ch: dict) -> bool:
    challenger = ch.get("challenger") or {}
    return str(challenger.get("id", "")).lower() == BOT_ID


print("✅ Bot online – waiting for casual-standard challenges …", flush=True)

for evt in client.bots.stream_incoming_events():
    if evt["type"] == "challenge":
        ch = evt["challenge"]
        if is_from_me(ch):
            continue
        if ch["variant"]["key"] == "standard" and not ch["rated"]:
            client.challenges.accept(ch["id"])
        else:
            client.challenges.decline(ch["id"])

    elif evt["type"] == "gameStart":
        threading.Thread(target=play_game,
                         args=(evt["game"]["id"],),
                         daemon=True).start()
