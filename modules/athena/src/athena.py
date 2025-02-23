import chess
import numpy as np
import tensorflow as tf
import json
from pathlib import Path
from modules.athena.src.utils import fen_to_tensor, move_to_index
from modules.athena.src.dqn import DQN

class Athena:
    def __init__(self, model_path, move_vocab_path):
        """Initialize Athena with the trained model and move vocabulary."""
        self.model = tf.keras.models.load_model(
            model_path, 
            custom_objects={"DQN": DQN}
        )

        with open(move_vocab_path, "r") as f:
            self.move_vocab = json.load(f)

    def predict_move(self, fen, move_history, legal_moves_mask, turn_indicator):
        """Predicts the best move using Athena's neural network."""
        
        # Convert inputs
        fen_tensor = np.expand_dims(fen_to_tensor(fen), axis=0)
        move_history_encoded = (
            np.array([move_to_index(move_history, self.move_vocab)]).reshape(1, -1)
            if move_history
            else np.zeros((1, 50))
        )
        legal_moves_array = np.expand_dims(legal_moves_mask, axis=0)
        turn_indicator_array = np.array([[turn_indicator]])  # 1 for white, 0 for black

        # Model Prediction
        move_output = self.model.predict(
            [fen_tensor, move_history_encoded, legal_moves_array, turn_indicator_array]
        )

        move_index = np.argmax(move_output[0])
        legal_moves_list = list(chess.Board(fen).legal_moves)

        if move_index < len(legal_moves_list):
            best_move = legal_moves_list[move_index]
        else:
            best_move = legal_moves_list[0]  # Fallback move

        print(f"♟️ Athena Move: {best_move}")
        return best_move
