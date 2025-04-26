import sys
from pathlib import Path
import chess
import logging
import numpy as np
import tensorflow as tf
from collections import deque
from typing import List, Dict, Any
from tqdm import tqdm

# Dynamically set the root path to the "Kronos" directory
ROOT_PATH = Path(__file__).resolve().parents[3]  # Moves up to "Kronos/"
sys.path.append(str(ROOT_PATH))
from modules.athena.src.utils import fen_to_tensor, encode_move_sequence, get_attack_defense_maps

logger = logging.getLogger('athena.selfplay')

class SelfPlayTrainer:
    def __init__(self, athena, buffer_size=500000):
        self.athena = athena
        self.game_memory = deque(maxlen=buffer_size)
        self.batch_size = 256
        self.num_epochs = 1
        self.games_per_iteration = 1
        
    def train_iteration(self, iteration: int) -> Dict[str, Any]:
        """Run one iteration of self-play training."""
        logger.info(f"Starting self-play iteration {iteration}")
        
        # Generate games through self-play
        games_data = []
        pbar = tqdm(total=self.games_per_iteration, desc="Playing games")
        
        for game_id in range(self.games_per_iteration):
            game_states = self._play_game()
            if game_states:
                games_data.extend(game_states)
                pbar.update(1)
                pbar.set_postfix({'positions': len(game_states)})
        
        pbar.close()
        
        # Add new games to memory
        self.game_memory.extend(games_data)
        logger.info(f"Memory buffer size: {len(self.game_memory)}")
        
        # Train on sampled positions
        metrics = self._train_on_memory()
        
        # Add some additional metrics
        metrics['game_positions'] = len(games_data)
        metrics['buffer_size'] = len(self.game_memory)
        
        return metrics
    
    def _play_game(self) -> List[Dict]:
        """Play a single game of self-play."""
        try:
            board = chess.Board()
            game_states = []
            move_count = 0
            max_moves = 200  # Prevent infinite games
            
            while not board.is_game_over() and move_count < max_moves:
                # Get current state
                state = {
                    'board': fen_to_tensor(board),
                    'history': encode_move_sequence([m.uci() for m in board.move_stack]),
                    'attack_map': get_attack_defense_maps(board)[0],
                    'defense_map': get_attack_defense_maps(board)[1]
                }
                
                # Get move probabilities from MCTS
                move_probs = self.athena.mcts.run(board)
                
                if not move_probs:
                    break
                    
                # Create policy target
                policy = np.zeros(4096)  # 64*64 possible moves
                for move, prob in move_probs.items():
                    policy[move.from_square * 64 + move.to_square] = prob
                
                # Store position
                game_states.append({
                    'state': state,
                    'policy': policy,
                    'turn': board.turn
                })
                
                # Choose move (with temperature)
                temperature = 1.0 if move_count < 30 else 0.5  # Reduce temperature later in game
                if temperature == 0:
                    # Select best move deterministically
                    move = max(move_probs.items(), key=lambda x: x[1])[0]
                else:
                    # Sample move based on probabilities
                    moves, probs = zip(*move_probs.items())
                    probs = np.array(probs) ** (1 / temperature)
                    probs /= np.sum(probs)
                    move = np.random.choice(moves, p=probs)
                
                board.push(move)
                move_count += 1
            
            # Get game result
            if board.is_checkmate():
                result = 1 if not board.turn else -1
            elif board.is_stalemate() or board.is_insufficient_material():
                result = 0
            else:
                result = 0
            
            # Add value targets
            for state in game_states:
                state['value'] = result if state['turn'] else -result
            
            return game_states
            
        except Exception as e:
            logger.error(f"Error in self-play game: {str(e)}")
            return None

    def _train_on_memory(self) -> Dict[str, float]:
        """Train on sampled positions from memory."""
        if len(self.game_memory) < self.batch_size:
            logger.warning(f"Not enough samples for training: {len(self.game_memory)} < {self.batch_size}")
            return {}
            
        logger.info("Training on memory buffer")
        metrics_history = []
        
        try:
            for epoch in range(self.num_epochs):
                # Sample batch
                indices = np.random.choice(len(self.game_memory), size=self.batch_size, replace=False)
                
                # Prepare training data
                states = []
                policies = []
                values = []
                
                for idx in indices:
                    sample = self.game_memory[idx]
                    states.append(sample['state'])
                    policies.append(sample['policy'])
                    values.append(sample.get('value', 0.0))
                
                # Convert to arrays
                states = {k: np.array([s[k] for s in states]) for k in states[0].keys()}
                policies = np.array(policies)
                values = np.array(values)
                
                # Train step
                metrics = self.athena.train_step(states, policies, values)
                metrics_history.append(metrics)
                
                logger.info(f"Epoch {epoch + 1}/{self.num_epochs} - " + 
                          " - ".join(f"{k}: {v:.4f}" for k, v in metrics.items()))
            
            # Average metrics across epochs
            avg_metrics = {}
            for key in metrics_history[0].keys():
                avg_metrics[key] = np.mean([m[key] for m in metrics_history])
            
            return avg_metrics
            
        except Exception as e:
            logger.error(f"Error in training: {str(e)}")
            return {}