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
ROOT_PATH = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_PATH))
from modules.athena.src.utils import fen_to_tensor, encode_move_sequence, get_attack_defense_maps
from modules.ares.logic.mcts import MCTS

logger = logging.getLogger('athena.selfplay')

class SelfPlayTrainer:
    def __init__(self, athena, buffer_size=500000):
        self.athena = athena
        self.game_memory = deque(maxlen=buffer_size)
        self.batch_size = 512
        self.num_epochs = 1
        self.games_per_iteration = 10
        self.num_simulations = 25  # MCTS simulations

    def train_iteration(self, iteration: int) -> Dict[str, Any]:
        logger.info(f"[SelfPlayTrainer] Starting self-play iteration {iteration}")
        games_data = []
        pbar = tqdm(total=self.games_per_iteration, desc="[Self-Play] Games")

        for game_id in range(self.games_per_iteration):
            logger.info(f"[Self-Play] Playing Game {game_id+1}/{self.games_per_iteration}")
            game_states = self._play_game()
            if game_states:
                games_data.extend(game_states)
                pbar.update(1)

        pbar.close()

        self.game_memory.extend(games_data)
        logger.info(f"[SelfPlayTrainer] Memory buffer size: {len(self.game_memory)}")

        metrics = self._train_on_memory()
        metrics['game_positions'] = len(games_data)
        metrics['buffer_size'] = len(self.game_memory)

        return metrics

    def _play_game(self) -> List[Dict]:
        try:
            board = chess.Board()
            game_states = []
            move_count = 0
            max_moves = 200  # Prevent infinite games

            mcts = MCTS(network=self.athena, num_simulations=self.num_simulations)

            while not board.is_game_over() and move_count < max_moves:
                # Run MCTS
                move_probs = mcts.run(board)

                if not move_probs:
                    break

                # Store state
                state = {
                    'board': fen_to_tensor(board),
                    'history': encode_move_sequence([m.uci() for m in board.move_stack]),
                    'attack_map': get_attack_defense_maps(board)[0],
                    'defense_map': get_attack_defense_maps(board)[1]
                }

                policy = np.zeros(4096)
                for move, prob in move_probs.items():
                    policy[move.from_square * 64 + move.to_square] = prob

                game_states.append({
                    'state': state,
                    'policy': policy,
                    'turn': board.turn
                })

                # Move selection (temperature)
                temperature = 1.0 if move_count < 30 else 0.5
                moves, probs = zip(*move_probs.items())
                probs = np.array(probs)

                if temperature == 0:
                    move = moves[np.argmax(probs)]
                else:
                    probs = probs ** (1 / temperature)
                    probs /= np.sum(probs)
                    move = np.random.choice(moves, p=probs)

                board.push(move)
                move_count += 1

            # Assign value targets
            if board.is_checkmate():
                result = 1 if not board.turn else -1
            elif board.is_stalemate() or board.is_insufficient_material():
                result = 0
            else:
                result = 0

            for state in game_states:
                state['value'] = result if state['turn'] else -result

            return game_states

        except Exception as e:
            logger.error(f"Error during self-play game: {str(e)}")
            return None

    def _train_on_memory(self) -> Dict[str, float]:
        if len(self.game_memory) < self.batch_size:
            logger.warning(f"Not enough samples to train: {len(self.game_memory)}")
            return {}

        logger.info("Training on memory buffer")
        metrics_history = []

        try:
            for epoch in range(self.num_epochs):
                indices = np.random.choice(len(self.game_memory), size=self.batch_size, replace=False)

                states = []
                policies = []
                values = []

                for idx in indices:
                    sample = self.game_memory[idx]
                    states.append(sample['state'])
                    policies.append(sample['policy'])
                    values.append(sample['value'])

                states = {k: np.array([s[k] for s in states]) for k in states[0].keys()}
                policies = np.array(policies)
                values = np.array(values)

                metrics = self.athena.train_step(states, policies, values)
                metrics_history.append(metrics)

                logger.info(f"Epoch {epoch+1}/{self.num_epochs} - " + " - ".join(f"{k}: {v:.4f}" for k, v in metrics.items()))

            avg_metrics = {}
            for key in metrics_history[0].keys():
                avg_metrics[key] = np.mean([m[key] for m in metrics_history])

            return avg_metrics

        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            return {}
