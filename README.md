# Chess-Bot (Random-Move Prototype)

A minimal Lichess BOT that accepts **casual ▸ standard** challenges and
replies with totally random legal moves.  
Built for learning + experimentation rather than strong play.
This should work out of the box.

1. Gemini Model 
 - Messed up on castling ??? 
 - made some ill fated knight jumps
 - still takes a while to make moves
 - blundered mate in 1. 

---

## Quick start

```bash
# 1 – clone the repo & create a virtual-env
git clone https://github.com/your-user/chess-bot.git
cd chess-bot
python3 -m venv .venv
source .venv/bin/activate      # .venv\Scripts\activate on Windows

# 2 – install deps
pip install berserk python-chess

# 3 – set your Lichess API token (must have scopes: bot:play, challenge:read, challenge:write)
export LICHESS_TOKEN=pa_xxxxxxxxxxxxxxxxx

# 4 – run the bot
python bot.py


