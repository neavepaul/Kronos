import sys
from pathlib import Path
import chess
import chess.engine
import numpy as np
import random
from typing import Dict, Any, Optional, List
from tqdm import tqdm
import tensorflow as tf

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

        self.engine_white.configure({'Skill Level': 5, 'Threads': 1, 'Hash': 256})
        self.engine_black.configure({'Skill Level': 0, 'Threads': 1, 'Hash': 256})

        self.num_games = 20
        self.max_game_length = 160

    def train_from_stronger_stockfish(self) -> Dict[str, Any]:
        print("\n[StockfishDualTrainer] Lv5 vs Lv0 Training Mode (PrometheusNet-Compatible)")
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
                    scaled_value = np.tanh(cp_score / 400.0)

                    soft_policy = np.zeros((64, 64), dtype=np.float32)
                    scores = []
                    moves = []

                    for info in eval_info:
                        move_i = info.get("pv", [])[0] if info.get("pv") else None
                        score_i = info['score'].white().score(mate_score=10000)
                        if move_i is not None and score_i is not None:
                            moves.append(move_i)
                            scores.append(score_i)

                    if moves and scores:
                        scores_np = np.array(scores, dtype=np.float32)
                        temp = 150.0
                        scaled = scores_np / temp
                        scaled -= np.max(scaled)
                        exp_scores = np.exp(scaled)
                        probs = exp_scores / np.sum(exp_scores)

                        for move_i, prob in zip(moves, probs):
                            soft_policy[move_i.from_square, move_i.to_square] = prob

                        top_move = moves[0]
                        soft_policy *= 0.8
                        soft_policy[top_move.from_square, top_move.to_square] += 0.2
                        soft_policy /= np.sum(soft_policy)

                        epsilon = 1e-5
                        soft_policy = (1 - epsilon) * soft_policy + epsilon / (64 * 64)

                        # Determine promotion class
                        if top_move.promotion is None:
                            promo_class = 0
                        elif top_move.promotion == chess.QUEEN:
                            promo_class = 1
                        elif top_move.promotion == chess.ROOK:
                            promo_class = 2
                        elif top_move.promotion == chess.BISHOP:
                            promo_class = 3
                        elif top_move.promotion == chess.KNIGHT:
                            promo_class = 4
                        else:
                            promo_class = 0

                        state = self._encode_state(board)
                        positions.append({
                            'state': state,
                            'policy': soft_policy,
                            'promotion': tf.keras.utils.to_categorical(promo_class, num_classes=5),
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
        boards, histories, attack_maps, defense_maps = [], [], [], []
        policies, promotions, values = [], [], []

        for pos in positions:
            s = pos['state']
            boards.append(s['board'])
            histories.append(s['history'])
            attack_maps.append(s['attack_map'])
            defense_maps.append(s['defense_map'])
            policies.append(pos['policy'])
            promotions.append(pos['promotion'])
            values.append(pos['value'])

        board_arr = np.array(boards)
        history_arr = np.array(histories)
        attack_arr = np.array(attack_maps)
        defense_arr = np.array(defense_maps)
        policy_arr = np.array(policies)
        promotion_arr = np.array(promotions)
        value_arr = np.array(values)

        print("[Trainer] Training on collected positions...")
        losses = self.athena.model.train_on_batch(
            [board_arr, history_arr, attack_arr, defense_arr],
            [policy_arr, promotion_arr, value_arr]
        )

        return {
            'total_loss': float(losses[0]),
            'policy_loss': float(losses[1]),
            'promotion_loss': float(losses[2]),
            'value_loss': float(losses[3])
        }

    def __del__(self):
        self.engine_white.quit()
        self.engine_black.quit()
