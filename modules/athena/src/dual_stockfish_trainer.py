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
        self.num_games = 10
        self.max_game_length = 160

    def train_from_stronger_stockfish(self) -> Dict[str, Any]:
        print("\n[StockfishDualTrainer] Lv2 vs Lv1 Training Mode")
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
                # Evaluate the board using Stockfish for value
                try:
                    eval_info = engine.analyse(board, chess.engine.Limit(depth=10))
                    cp_score = eval_info['score'].white().score(mate_score=10000)
                    clamped = max(min(cp_score, 1000), -1000)
                    scaled_value = clamped / 1000.0
                except Exception as e:
                    print(f"[Eval Error] {e}, fallback to 0.0")
                    scaled_value = 0.0

                state = self._encode_state(board)
                policy = self._encode_policy(move)
                positions.append({
                    'state': state,
                    'policy': policy,
                    'value': scaled_value
                })

            board.push(move)

        return positions

    def _encode_state(self, board: chess.Board) -> Dict[str, np.ndarray]:
        return {
            'board': fen_to_tensor(board),
            'history': encode_move_sequence([m.uci() for m in board.move_stack]),
            'attack_map': get_attack_defense_maps(board)[0],
            'defense_map': get_attack_defense_maps(board)[1]
        }

    def _encode_policy(self, move: chess.Move) -> np.ndarray:
        policy = np.zeros(4096, dtype=np.float32)
        policy[move.from_square * 64 + move.to_square] = 1.0
        return policy

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
