import sys
from pathlib import Path
import chess
import chess.engine
import numpy as np
import random
import tensorflow as tf
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from tqdm import tqdm

# Dynamically set the root path to the "Kronos" directory
ROOT_PATH = Path(__file__).resolve().parents[3]  # Moves up to "Kronos/"
sys.path.append(str(ROOT_PATH))
from modules.athena.src.utils import fen_to_tensor, encode_move_sequence, get_attack_defense_maps
from modules.athena.src.evaluate import quick_evaluate_model

class StockfishTrainer:
    def __init__(self, athena, stockfish_path: Optional[str] = None, depth: int = 15):
        if stockfish_path is None:
            stockfish_path = str(Path(__file__).parent.parent.parent / 'shared' / 'stockfish' / 'stockfish-windows-x86-64-avx2.exe')
        
        self.athena = athena
        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        self.engine.configure({'Threads': 4, 'Hash': 1024, 'Skill Level': 20})
        self.depth = depth
        print(f"[StockfishTrainer] Initialized Stockfish at {stockfish_path}")
        self.batch_size = 256
        self.num_games = 10
        self.evaluation_games = 6
        self.max_game_length = 160
        self.tablebase_piece_count = 7
        self.endgame_piece_count = 12
        
    def _count_pieces(self, board: chess.Board) -> int:
        """Count total pieces on the board."""
        return sum(len(board.pieces(piece_type, color))
                  for color in [chess.WHITE, chess.BLACK]
                  for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP,
                                   chess.ROOK, chess.QUEEN, chess.KING])

    def train_iteration(self, iteration: int, run_evaluation: bool = False) -> Dict[str, Any]:
        print(f"\n[StockfishTrainer] Iteration {iteration} (Run Evaluation: {run_evaluation})")
        total_metrics = {'policy_loss': 0.0, 'value_loss': 0.0, 'total_loss': 0.0, 'games_processed': 0, 'avg_game_length': 0}
        
        for game_num in range(self.num_games):
            print(f"[Stockfish] Starting Game {game_num+1}/{self.num_games}")
            positions = self._generate_game_positions()
            print(f"[Stockfish] Game {game_num+1} finished with {len(positions)} positions")
            total_metrics['avg_game_length'] += len(positions)

            for i in range(0, len(positions), self.batch_size):
                batch = positions[i:i + self.batch_size]
                metrics = self._train_on_positions(batch)
                for k, v in metrics.items():
                    if k in total_metrics:
                        total_metrics[k] += v

            total_metrics['games_processed'] += 1

        games = total_metrics['games_processed']
        metrics_avg = {k: total_metrics[k] / games for k in ['policy_loss','value_loss','total_loss','avg_game_length']}

        elo_estimate = quick_evaluate_model(self.athena) if run_evaluation else 0.0

        return {**metrics_avg, 'elo_estimate': elo_estimate}

    def _generate_game_positions(self) -> List[Dict]:
        """Generate positions from a fast game."""
        positions = []
        board = chess.Board()
        initial_material = self._count_pieces(board)

        while not board.is_game_over() and len(board.move_stack) < self.max_game_length * 2:
            curr_depth = 6
            time_per_move = 0.05

            move = self.engine.play(board, chess.engine.Limit(time=time_per_move)).move
            
            if move is None:
                break

            state_tensor = fen_to_tensor(board)
            move_history = encode_move_sequence([m.uci() for m in board.move_stack])
            attack_map, defense_map = get_attack_defense_maps(board)
            
            policy = np.zeros(4096)
            policy[move.from_square * 64 + move.to_square] = 1.0

            positions.append({
                'state': {
                    'board': state_tensor,
                    'history': move_history,
                    'attack_map': attack_map,
                    'defense_map': defense_map
                },
                'policy': policy,
                'value': 0.0,  # We'll adjust values after game ends
                'metadata': {}
            })
            
            board.push(move)

        result = board.result()
        final_value = 1.0 if result == "1-0" else (-1.0 if result == "0-1" else 0.0)
        
        for pos in positions:
            pos['value'] = final_value

        return positions

    def _train_on_positions(self, positions: List[Dict]) -> Dict[str, float]:
        """Train on a batch of positions using the compiled alpha_model."""
        boards, histories, attack_maps, defense_maps, policies, values = [], [], [], [], [], []
        for pos in positions:
            state = pos['state']
            boards.append(state['board'])
            histories.append(state['history'])
            attack_maps.append(state['attack_map'])
            defense_maps.append(state['defense_map'])
            policies.append(pos['policy'])
            values.append(pos['value'])
        board_arr = np.array(boards)
        history_arr = np.array(histories)
        attack_arr = np.array(attack_maps)
        defense_arr = np.array(defense_maps)
        policy_arr = np.array(policies)
        value_arr = np.array(values)
        losses = self.athena.alpha_model.train_on_batch([
            board_arr, history_arr, attack_arr, defense_arr
        ], [policy_arr, value_arr])
        return {'total_loss': float(losses[0]), 'policy_loss': float(losses[1]), 'value_loss': float(losses[2])}


    def _evaluate_against_stockfish(self) -> Dict[str, float]:
        """Play games against Stockfish to measure performance, with a max move limit and no per-move analysis."""
        print(f"Evaluating against Stockfish ({self.evaluation_games} games)...")
        wins = draws = losses = 0
        max_moves = self.max_game_length  # avoid endless games

        for _ in range(self.evaluation_games):
            board = chess.Board()
            athena_is_white = bool(random.getrandbits(1))
            move_count = 0

            # simulate until game over or move limit reached
            while not board.is_game_over() and move_count < max_moves:
                if (board.turn == chess.WHITE) == athena_is_white:
                    # Athena's turn: pick model's top policy move
                    move_probs, _ = self.athena.predict(board)
                    move = max(move_probs, key=move_probs.get)
                else:
                    # Stockfish's turn: use engine.play
                    result = self.engine.play(board, chess.engine.Limit(depth=self.depth))
                    move = result.move

                # ensure legal
                if move not in board.legal_moves:
                    move = random.choice(list(board.legal_moves))

                board.push(move)
                move_count += 1

            result = board.result()
            if result == "1-0":
                wins += 1 if athena_is_white else 0
                losses += 0 if athena_is_white else 1
            elif result == "0-1":
                wins += 1 if not athena_is_white else 0
                losses += 0 if not athena_is_white else 1
            else:
                draws += 1

        games_played = wins + draws + losses
        win_rate = wins / games_played if games_played else 0.0
        draw_rate = draws / games_played if games_played else 0.0

        return {
            'win_rate': win_rate,
            'draw_rate': draw_rate,
        }

    def __del__(self):
        if hasattr(self, 'engine'):
            self.engine.quit()