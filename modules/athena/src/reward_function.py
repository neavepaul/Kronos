import numpy as np
import chess

# Standard Piece Values
PIECE_VALUES = {"p": 1, "n": 3, "b": 3, "r": 5, "q": 9, "k": 0}  # King has no material value

# Reward Scaling Factors
MATERIAL_WEIGHT = 0.1  # Material shaping should not overpower the final win/loss reward
OUTCOME_REWARD = 1.0    # Winning should be the highest priority
DRAW_PENALTY = -0.5     # Penalize stalemates to avoid Athena trading everything into draws

def compute_reward(board_before, board_after, athena_move, game_result):
    """
    Computes Athena's reinforcement learning reward while adjusting for color.

    Parameters:
        board_before (chess.Board): Board state before Athena's move.
        board_after (chess.Board): Board state after Athena's move.
        athena_move (str): The move played by Athena in UCI format.
        game_result (int or None): +1 if Athena wins, -1 if it loses, 0 if draw (only at game end).

    Returns:
        float: Computed reward.
    """

    # Determine Athena's color
    color_factor = 1 if board_before.turn == chess.WHITE else -1  # White → +1, Black → -1

    # Compute Material Balance Before and After the Move
    material_before = sum(PIECE_VALUES[p.symbol().lower()] for p in board_before.piece_map().values())
    material_after = sum(PIECE_VALUES[p.symbol().lower()] for p in board_after.piece_map().values())

    # Reward for Material Gain (Encourage Capturing Opponent’s Pieces)
    material_reward = (material_after - material_before) * color_factor * MATERIAL_WEIGHT

    # Penalize Losing Own Pieces (Encourage Material Protection)
    if board_before.is_capture(athena_move):
        piece_captured = board_before.piece_at(chess.parse_square(athena_move[2:4]))
        if piece_captured:
            material_reward += PIECE_VALUES[piece_captured.symbol().lower()] * color_factor * MATERIAL_WEIGHT

    # Final Game Result Reward
    final_reward = 0
    if game_result is not None:
        if game_result == 0:  # Draw (Stalemate or repetition)
            final_reward = DRAW_PENALTY
        else:
            final_reward = game_result * OUTCOME_REWARD * color_factor

    # Total Reward
    return material_reward + final_reward
