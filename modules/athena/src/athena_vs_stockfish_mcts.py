import chess
import chess.engine
import random
import sys
import os
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path

# Load Athena model
ROOT_PATH = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_PATH))
from modules.athena.src.aegis_net import AegisNet
from modules.athena.src.utils import fen_to_tensor, encode_move_sequence, get_attack_defense_maps
from modules.ares.logic.mcts import MCTS

MODEL_PATH = str(ROOT_PATH / "modules/athena/src/models/athena_hybrid_final_20250427_112111_elo_1800.weights.h5")
STOCKFISH_PATH = str(ROOT_PATH / "modules/shared/stockfish/stockfish-windows-x86-64-avx2.exe")

model = AegisNet()
model.alpha_model.load_weights(MODEL_PATH)
ares_mcts = MCTS(network=model, num_simulations=25)


# --- Config ---
NUM_GAMES = 50
LEVEL_CHANGE_EVERY = 10
START_LEVEL = 5
MAX_LEVEL = 20

BLUNDER_THRESHOLD = 150
MISTAKE_THRESHOLD = 100
INACCURACY_THRESHOLD = 50

# Initialize Stockfish
engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

# Report tracking
report = {
    'total': {'wins': 0, 'losses': 0, 'draws': 0},
    'white': {'wins': 0, 'losses': 0, 'draws': 0},
    'black': {'wins': 0, 'losses': 0, 'draws': 0},
    'wdl_by_level': {},
    'mistakes': [],
    'total_moves': 0
}

# --- Helper functions ---
def model_move(board):
    fen_tensor = np.expand_dims(fen_to_tensor(board), axis=0)
    move_history = encode_move_sequence([m.uci() for m in board.move_stack])
    move_history = np.expand_dims(move_history, axis=0)
    attack_map, defense_map = get_attack_defense_maps(board)
    attack_map = np.expand_dims(attack_map, axis=0)
    defense_map = np.expand_dims(defense_map, axis=0)

    policy, _ = model.alpha_model.predict([fen_tensor, move_history, attack_map, defense_map], verbose=0)
    legal_moves = list(board.legal_moves)

    move_probs = {}
    for move in legal_moves:
        idx = move.from_square * 64 + move.to_square
        move_probs[move] = policy[0][idx]

    if move_probs:
        return max(move_probs.items(), key=lambda x: x[1])[0]
    else:
        return random.choice(list(board.legal_moves))


def mcts_move(board):
    move_probs = ares_mcts.run(board)
    if move_probs:
        return max(move_probs.items(), key=lambda x: x[1])[0]
    else:
        return random.choice(list(board.legal_moves))


# --- Main test loop ---
current_level = START_LEVEL
engine.configure({'Skill Level': current_level})

for game_num in range(1, NUM_GAMES + 1):
    board = chess.Board()
    play_as_white = random.choice([True, False])
    player_color = chess.WHITE if play_as_white else chess.BLACK

    while not board.is_game_over():
        if board.turn == player_color:
            move = mcts_move(board)
        else:
            result = engine.play(board, chess.engine.Limit(time=0.05))
            move = result.move

        info_before = engine.analyse(board, chess.engine.Limit(depth=10))
        eval_before = info_before['score'].white().score(mate_score=10000)

        board.push(move)

        if not board.is_game_over():
            info_after = engine.analyse(board, chess.engine.Limit(depth=10))
            eval_after = info_after['score'].white().score(mate_score=10000)

            if eval_before is not None and eval_after is not None:
                eval_drop = eval_before - eval_after
                if board.turn != player_color and eval_drop > INACCURACY_THRESHOLD:
                    severity = ""
                    if eval_drop > BLUNDER_THRESHOLD:
                        severity = "blunder"
                    elif eval_drop > MISTAKE_THRESHOLD:
                        severity = "mistake"
                    else:
                        severity = "inaccuracy"
                    report['mistakes'].append({
                        'fen': board.fen(),
                        'move': move.uci(),
                        'drop': eval_drop,
                        'severity': severity
                    })

        report['total_moves'] += 1

    # --- Game result ---
    result = board.result()
    level_key = f"Level {current_level}"
    if level_key not in report['wdl_by_level']:
        report['wdl_by_level'][level_key] = {'wins': 0, 'losses': 0, 'draws': 0}

    if result == '1-0':
        if player_color == chess.WHITE:
            report['white']['wins'] += 1
            report['total']['wins'] += 1
            report['wdl_by_level'][level_key]['wins'] += 1
        else:
            report['black']['losses'] += 1
            report['total']['losses'] += 1
            report['wdl_by_level'][level_key]['losses'] += 1
    elif result == '0-1':
        if player_color == chess.BLACK:
            report['black']['wins'] += 1
            report['total']['wins'] += 1
            report['wdl_by_level'][level_key]['wins'] += 1
        else:
            report['white']['losses'] += 1
            report['total']['losses'] += 1
            report['wdl_by_level'][level_key]['losses'] += 1
    else:
        if player_color == chess.WHITE:
            report['white']['draws'] += 1
        else:
            report['black']['draws'] += 1
        report['total']['draws'] += 1
        report['wdl_by_level'][level_key]['draws'] += 1

    if game_num % LEVEL_CHANGE_EVERY == 0 and current_level < MAX_LEVEL:
        current_level += 5
        engine.configure({'Skill Level': min(current_level, MAX_LEVEL)})
        print(f"\n[INFO] Increasing Stockfish Skill Level to {current_level}\n")

# --- Save mistakes ---
with open("mistakes_report_mcts.json", "w") as f:
    json.dump(report['mistakes'], f, indent=2)

# --- Plot W/D/L ---
levels = sorted(report['wdl_by_level'].keys(), key=lambda x: int(x.split()[1]))
wins = [report['wdl_by_level'][lvl]['wins'] for lvl in levels]
losses = [report['wdl_by_level'][lvl]['losses'] for lvl in levels]
draws = [report['wdl_by_level'][lvl]['draws'] for lvl in levels]

x_vals = [int(lvl.split()[1]) for lvl in levels]
total_per_level = [wins[i] + losses[i] + draws[i] for i in range(len(wins))]

win_pct = [100 * wins[i] / total_per_level[i] for i in range(len(wins))]
loss_pct = [100 * losses[i] / total_per_level[i] for i in range(len(wins))]
draw_pct = [100 * draws[i] / total_per_level[i] for i in range(len(wins))]

plt.figure(figsize=(10, 6))
plt.plot(x_vals, win_pct, label='Win %', marker='o')
plt.plot(x_vals, draw_pct, label='Draw %', marker='o')
plt.plot(x_vals, loss_pct, label='Loss %', marker='o')
plt.xlabel("Stockfish Skill Level")
plt.ylabel("Percentage")
plt.title("Athena vs Stockfish: Performance over Skill Levels")
plt.legend()
plt.grid(True)
plt.savefig("performance_graph_mcts.png")
plt.show()

# --- Final report ---
print("\n================ Test Report ================")
print(f"Games Played: {NUM_GAMES}")
print(f"\nTotal Results:")
print(f" Wins: {report['total']['wins']}")
print(f" Losses: {report['total']['losses']}")
print(f" Draws: {report['total']['draws']}")
print(f"\nWhite Side Results:")
print(f" Wins: {report['white']['wins']}")
print(f" Losses: {report['white']['losses']}")
print(f" Draws: {report['white']['draws']}")
print(f"\nBlack Side Results:")
print(f" Wins: {report['black']['wins']}")
print(f" Losses: {report['black']['losses']}")
print(f" Draws: {report['black']['draws']}")
print(f"\nTotal Moves: {report['total_moves']}")
print(f"Mistakes/Inaccuracies/Blunders logged: {len(report['mistakes'])}")
print("============================================")

engine.quit()
