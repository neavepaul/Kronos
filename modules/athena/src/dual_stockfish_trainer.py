import sys
from pathlib import Path
import chess
import chess.engine
import numpy as np
import random
from typing import Dict, Any, Optional, List
from tqdm import tqdm

# Dynamically set the root path to the "Kronos" directory
ROOT_PATH = Path(__file__).resolve().parents[3]  # Up to "Kronos/"
sys.path.append(str(ROOT_PATH))
from modules.athena.src.utils import fen_to_tensor, encode_move_sequence, get_attack_defense_maps

class StockfishDualTrainer:
    def __init__(self, athena, stockfish_path: Optional[str] = None):
        if stockfish_path is None:
            stockfish_path = str(Path(__file__).parent.parent.parent / 'shared' / 'stockfish' / 'stockfish-windows-x86-64-avx2.exe')

        self.athena = athena
        self.engine_white = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        self.engine_black = chess.engine.SimpleEngine.popen_uci(stockfish_path)

        self.engine_white.configure({'Skill Level': 2, 'Threads': 1, 'Hash': 256})
        self.engine_black.configure({'Skill Level': 1, 'Threads': 1, 'Hash': 256})

        self.batch_size = 256
        self.num_games = 20
        self.max_game_length = 160

    def train_from_stronger_stockfish(self) -> Dict[str, Any]:
        print("\n[StockfishDualTrainer] Lv2 vs Lv1 Training Mode (Soft Targets + Eval + Label Smoothing + Top Weight)")
        all_positions = []

        for game_id in range(self.num_games):
            print(f"[StockfishDualTrainer] Game {game_id + 1}/{self.num_games}")
            game_data = self._play_stockfish_vs_stockfish()
            if game_data:
                all_positions.extend(game_data)

        print(f"[StockfishDualTrainer] Total positions collected: {len(all_positions)}")

        metrics = self._train_on_positions(all_positions)
        metrics['games_processed'] = self.num_games
        return metrics

    def _play_stockfish_vs_stockfish(self) -> List[Dict[str, Any]]:
        board = chess.Board()
        positions = []

        is_lv2_white = bool(random.getrandbits(1))
        engine_lv2 = self.engine_white if is_lv2_white else self.engine_black
        engine_lv1 = self.engine_black if is_lv2_white else self.engine_white

        while not board.is_game_over() and len(board.move_stack) < self.max_game_length:
            engine = engine_lv2 if board.turn == is_lv2_white else engine_lv1
            result = engine.play(board, chess.engine.Limit(time=0.05))
            move = result.move
            if move is None:
                break

            if board.turn == is_lv2_white:
                try:
                    eval_info = engine.analyse(board, chess.engine.Limit(depth=10), multipv=5)
                    cp_score = eval_info[0]['score'].white().score(mate_score=10000)
                    # Scale Stockfish evaluation to [-1, +1] using tanh.
                    # This smooths out extreme values (> ±400 centipawns), reduces gradient noise,
                    # and helps the model focus on positional quality rather than exact score magnitude.
                    scaled_value = np.tanh(cp_score / 400.0)

                    # # Optional: skip low-signal positions
                    # if abs(scaled_value) < 0.05:
                    #     board.push(move)
                    #     continue

                    soft_policy = np.zeros(4096, dtype=np.float32)
                    scores = []
                    moves = []
                    for info in eval_info:
                        move_i = info.get("pv", [])[0] if info.get("pv") else None
                        score_i = info['score'].white().score(mate_score=10000)
                        if move_i is not None and score_i is not None:
                            moves.append(move_i)
                            scores.append(score_i)

                    # Normalize scores using stable softmax
                    if moves and scores:
                        scores_np = np.array(scores, dtype=np.float32)
                        temp = 150.0
                        scaled = scores_np / temp
                        scaled -= np.max(scaled)  # stability trick
                        exp_scores = np.exp(scaled)
                        probs = exp_scores / np.sum(exp_scores)

                        for move_i, prob in zip(moves, probs):
                            idx = move_i.from_square * 64 + move_i.to_square
                            soft_policy[idx] = prob

                        # Reinject weight into top move
                        top_move_idx = moves[0].from_square * 64 + moves[0].to_square
                        soft_policy[top_move_idx] += 0.1
                        soft_policy /= np.sum(soft_policy)

                        # Minimal label smoothing
                        epsilon = 1e-5
                        soft_policy = (1 - epsilon) * soft_policy + epsilon / 4096

                        state = self._encode_state(board)
                        positions.append({
                            'state': state,
                            'policy': soft_policy,
                            'value': scaled_value
                        })
                except Exception as e:
                    print(f"[Eval Error] {e}, skipping position")

            board.push(move)

        return positions

    def _encode_state(self, board: chess.Board) -> Dict[str, np.ndarray]:
        return {
            'board': fen_to_tensor(board),
            'history': encode_move_sequence([m.uci() for m in board.move_stack]),
            'attack_map': get_attack_defense_maps(board)[0],
            'defense_map': get_attack_defense_maps(board)[1]
        }

    def _train_on_positions(self, positions: List[Dict]) -> Dict[str, float]:
        boards, histories, attack_maps, defense_maps, policies, values = [], [], [], [], [], []

        for pos in positions:
            s = pos['state']
            boards.append(s['board'])
            histories.append(s['history'])
            attack_maps.append(s['attack_map'])
            defense_maps.append(s['defense_map'])
            policies.append(pos['policy'])
            values.append(pos['value'])

        board_arr = np.array(boards)
        history_arr = np.array(histories)
        attack_arr = np.array(attack_maps)
        defense_arr = np.array(defense_maps)
        policy_arr = np.array(policies)
        value_arr = np.array(values)

        print("[Trainer] Training on collected positions...")
        losses = self.athena.alpha_model.train_on_batch([
            board_arr, history_arr, attack_arr, defense_arr
        ], [policy_arr, value_arr])

        return {
            'total_loss': float(losses[0]),
            'policy_loss': float(losses[1]),
            'value_loss': float(losses[2])
        }

    def __del__(self):
        self.engine_white.quit()
        self.engine_black.quit()
