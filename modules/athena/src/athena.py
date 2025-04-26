import chess
import numpy as np
import tensorflow as tf
import json
from pathlib import Path
from typing import Optional, Tuple, Dict

from modules.athena.src.utils import (
    fen_to_tensor, 
    encode_move_sequence, 
    get_attack_defense_maps,
    encode_move,
    decode_move
)
from modules.athena.src.alpha_net import AlphaNet
from modules.ares.logic.mcts import MCTS

class Athena:
    def __init__(self, model_path: Optional[str] = None, move_vocab_path: Optional[str] = None):
        """Initialize Athena with AlphaZero-style neural network and MCTS."""
        # Initialize neural network
        self.network = AlphaNet()
        
        # Compile model with appropriate losses and metrics
        self.network.alpha_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss={
                'policy_head': tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                'value_head': tf.keras.losses.MeanSquaredError()
            },
            loss_weights={
                'policy_head': 1.0,
                'value_head': 1.0
            },
            metrics={
                'policy_head': ['accuracy'],
                'value_head': ['mean_squared_error']
            }
        )
        
        if model_path:
            self.network.load_weights(model_path)
        
        # Initialize optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
            
        # Initialize MCTS
        self.mcts = MCTS(
            network=self.network,
            num_simulations=10,  # Reduced from 800 for faster testing
            c_puct=1.0  # Exploration constant
        )
        
        # Load move vocabulary if provided
        self.move_vocab = None
        if move_vocab_path:
            with open(move_vocab_path, "r") as f:
                self.move_vocab = json.load(f)
                
    def predict_move(self, board: chess.Board, move_history: list = None) -> chess.Move:
        """Predict best move using MCTS with neural network guidance."""
        # Get move probabilities from MCTS
        move_probs = self.mcts.run(board)
        
        # Select move with highest visit count (corresponds to strongest move)
        best_move = max(move_probs.items(), key=lambda x: x[1])[0]
        
        return best_move
    
    def train_step(self, states):
        """Train the neural network on a batch of data.
        
        Args:
            states: List of game state dictionaries containing:
                - state: Dict with board, history, attack_map, defense_map tensors
                - policy: Policy target vector 
                - value: Value target
        """
        # Extract components from state dictionaries
        board_tensors = tf.stack([state['state']['board'] for state in states])
        history_tensors = tf.stack([state['state']['history'] for state in states])
        attack_maps = tf.stack([state['state']['attack_map'] for state in states])
        defense_maps = tf.stack([state['state']['defense_map'] for state in states])
        
        # Get target values
        policies = tf.stack([state['policy'] for state in states])
        values = tf.stack([state['value'] for state in states])
        
        # Use the compiled model's train_on_batch method
        metrics = self.network.alpha_model.train_on_batch(
            [board_tensors, history_tensors, attack_maps, defense_maps],
            [policies, values]
        )
        
        return {
            'policy_loss': metrics[1],  # First loss is policy head
            'value_loss': metrics[2],   # Second loss is value head
            'total_loss': metrics[0]    # Total loss
        }
    
    def save_model(self, path: str):
        """Save the neural network model."""
        self.network.alpha_model.save_weights(path)
