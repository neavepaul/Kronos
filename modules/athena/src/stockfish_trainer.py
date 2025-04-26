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
from modules.athena.src.evaluate import evaluate_model

class StockfishTrainer:
    def __init__(self, athena, stockfish_path: Optional[str] = None, depth: int = 20):
        self.athena = athena
        if stockfish_path is None:
            # Default Stockfish location
            stockfish_path = str(Path(__file__).parent.parent.parent / 'shared' / 'stockfish' / 'stockfish-windows-x86-64-avx2.exe')
        
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
            self.engine.configure({
                'Threads': 8,
                'Hash': 1024,
                'Skill Level': 20
            })
            self.depth = depth
            print(f"Initialized Stockfish engine at {stockfish_path}")
        except Exception as e:
            print(f"Failed to initialize Stockfish: {e}")
            raise
        
        # Training parameters
        self.batch_size = 128  # Process positions in smaller batches
        self.num_games = 5    # Number of complete games per iteration
        self.evaluation_games = 1
        self.max_game_length = 200  # Maximum moves per game (400 plies)
        self.tablebase_piece_count = 7  # When to reduce analysis depth
        self.endgame_piece_count = 12   # When to consider it endgame
        
    def _count_pieces(self, board: chess.Board) -> int:
        """Count total pieces on the board."""
        return sum(len(board.pieces(piece_type, color))
                  for color in [chess.WHITE, chess.BLACK]
                  for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP,
                                   chess.ROOK, chess.QUEEN, chess.KING])

    def train_iteration(self, iteration: int) -> Dict[str, Any]:
        print(f"Starting Stockfish training iteration {iteration}")
        total_metrics = {'policy_loss': 0.0, 'value_loss': 0.0, 'total_loss': 0.0, 'games_processed': 0, 'avg_game_length': 0}
        for game_num in range(self.num_games):
            positions = self._generate_game_positions()
            total_metrics['avg_game_length'] += len(positions)
            for i in range(0, len(positions), self.batch_size):
                batch = positions[i:i + self.batch_size]
                metrics = self._train_on_positions(batch)
                for k, v in metrics.items():
                    if k in total_metrics:
                        total_metrics[k] += v
            total_metrics['games_processed'] += 1
        # compute averages
        games = total_metrics['games_processed']
        metrics_avg = {k: total_metrics[k] / games for k in ['policy_loss','value_loss','total_loss','avg_game_length']}
        # use evaluate.py for Elo estimate
        elo_estimate = evaluate_model(self.athena)
        return {**metrics_avg, 'elo_estimate': elo_estimate}

    def _generate_game_positions(self) -> List[Dict]:
        """Generate positions from a complete game."""
        positions = []
        board = chess.Board()
        
        # Track game phase metrics
        total_piece_count = self._count_pieces(board)
        initial_material = total_piece_count
        
        while not board.is_game_over() and len(board.move_stack) < self.max_game_length * 2:
            # Update piece count
            total_piece_count = self._count_pieces(board)
            material_ratio = total_piece_count / initial_material
            
            # Adjust search parameters based on game phase
            if total_piece_count <= self.tablebase_piece_count:
                # Near tablebase positions - reduce depth since Hades will handle these
                curr_depth = min(10, self.depth)
                time_per_move = 0.1  # Quick analysis for tablebase positions
            elif total_piece_count <= self.endgame_piece_count:
                # Endgame positions - maintain depth but increase time
                curr_depth = self.depth
                time_per_move = 0.5  # More time for critical endgame positions
            else:
                # Middlegame - normal analysis
                curr_depth = self.depth
                time_per_move = 0.25
            
            # Get Stockfish's evaluation and best move
            result = self.engine.analyse(
                board,
                chess.engine.Limit(depth=curr_depth, time=time_per_move)
            )
            
            best_move = result["pv"][0]
            
            # Get state representation
            state_tensor = fen_to_tensor(board)
            move_history = encode_move_sequence([m.uci() for m in board.move_stack])
            attack_map, defense_map = get_attack_defense_maps(board)
            
            # Create policy target from best move
            policy = np.zeros(4096)  # 64*64 possible moves
            policy[best_move.from_square * 64 + best_move.to_square] = 1.0
            
            # Get position value with phase-aware scaling
            score = result["score"].relative.score()
            if score is not None:
                # Scale evaluation based on game phase
                if total_piece_count <= self.endgame_piece_count:
                    # More sensitive evaluation in endgame
                    value = np.tanh(score / (500 * material_ratio))  
                else:
                    value = np.tanh(score / 1000)
            else:
                # Handle mate scores
                if result["score"].is_mate():
                    mate_score = result["score"].relative.mate()
                    value = 1.0 if mate_score > 0 else -1.0
                else:
                    value = 0.0
            
            # Store position with enhanced metadata
            positions.append({
                'state': {
                    'board': state_tensor,
                    'history': move_history,
                    'attack_map': attack_map,
                    'defense_map': defense_map
                },
                'policy': policy,
                'value': value,
                'metadata': {
                    'move_number': len(board.move_stack) // 2,
                    'piece_count': total_piece_count,
                    'material_ratio': material_ratio,
                    'phase': 'tablebase' if total_piece_count <= self.tablebase_piece_count 
                            else 'endgame' if total_piece_count <= self.endgame_piece_count 
                            else 'middlegame'
                }
            })
            
            # Make the move
            board.push(best_move)
        
        # If game ended naturally, update values with temporal decay
        if board.is_game_over():
            result = board.result()
            final_value = 1.0 if result == "1-0" else (-1.0 if result == "0-1" else 0.0)
            
            # Update position values with phase-aware decay
            for idx, pos in enumerate(positions):
                moves_from_end = len(positions) - idx
                # Stronger decay in endgame/tablebase positions
                if pos['metadata']['phase'] == 'tablebase':
                    decay_base = 0.98  # Strongest temporal connection
                elif pos['metadata']['phase'] == 'endgame':
                    decay_base = 0.96  # Strong temporal connection
                else:
                    decay_base = 0.94  # Normal decay for middlegame
                
                decay_factor = decay_base ** moves_from_end
                pos['value'] = pos['value'] * (1 - decay_factor) + final_value * decay_factor
        
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