import chess
import numpy as np
import tensorflow as tf
import json
from pathlib import Path
from modules.athena.src.model import get_model, TransformerBlock
from modules.athena.src.utils import fen_to_tensor, move_to_index

class Athena:
    def __init__(self, model_path, move_vocab_path):
        """Initialize Athena with the trained model and move vocabulary."""
        self.model = tf.keras.models.load_model(
            model_path, custom_objects={"TransformerBlock": TransformerBlock}
        )

        with open(move_vocab_path, "r") as f:
            self.move_vocab = json.load(f)

    def predict_move(self, board, move_history, eval_score):
        """Predicts the best move using Athena's neural network."""
        fen = board.fen()

        # Convert inputs
        fen_tensor = np.expand_dims(fen_to_tensor(fen), axis=0)
        move_history_encoded = (
            np.array([move_to_index(move_history, self.move_vocab)]).reshape(1, -1)
            if move_history
            else np.zeros((1, 50))
        )
        legal_moves_array = np.zeros((1, 64, 64))
        for move in board.legal_moves:
            legal_moves_array[0, move.from_square, move.to_square] = 1
        turn_indicator_array = np.array([[1 if board.turn == chess.WHITE else 0]])
        eval_score_array = np.array([[eval_score]])

        # Model Prediction
        move_output, _ = self.model.predict(
            [fen_tensor, move_history_encoded, legal_moves_array, turn_indicator_array, eval_score_array]
        )
        move_index = np.argmax(move_output[0])
        legal_moves_list = list(board.legal_moves)

        if move_index < len(legal_moves_list):
            best_move = legal_moves_list[move_index]
        else:
            best_move = legal_moves_list[0]  # Fallback move

        print(f"♟️ Athena Move: {best_move}")
        return best_move
