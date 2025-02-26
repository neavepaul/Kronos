import numpy as np
import chess
import requests

from modules.athena.src.game_memory import GameMemory
from modules.athena.src.utils import get_stockfish_eval_train

# Standard Piece Values
PIECE_VALUES = {"p": 1, "n": 3, "b": 3, "r": 5, "q": 9, "k": 0}  # King has no material value

# Major Rewards
OUTCOME_REWARD = 10.0   # Winning should be the highest priority
DRAW_PENALTY = -2.0     # Penalize draws to avoid Athena forcing unnecessary repetitions

# Medium Rewards
MATERIAL_WEIGHT = 0.1   # Material shaping should not overpower final game result
PROMOTION_REWARD = 5.0  # High reward for pawn promotion
PASSED_PAWN_REWARD = 1.0  # Encourages pawn promotion strategy
KING_SAFETY_PENALTY = -2.0  # Large penalty if king is exposed in midgame

# Minor Positional Rewards
CENTER_CONTROL_REWARD = 0.3  # Encourages occupying center
CASTLING_REWARD = 0.5   # Castling is good but shouldn’t be forced every game
CHECK_REWARD = 0.2      # Encourages attacking play
AVOID_REPETITION_PENALTY = -5.0  # Strong penalty for unnecessary repetition


def compute_reward(board_before, board_after, athena_move, game_result):
    """
    Computes Athena's reinforcement learning reward while adjusting for color.
    Uses Stockfish evaluations for position-based learning.
    """
    # Determine Athena's color
    color_factor = 1 if board_before.turn == chess.WHITE else -1  # White → +1, Black → -1

    # Fetch Stockfish evaluations
    eval_before = get_stockfish_eval_train(str(board_before.fen())) * color_factor
    eval_after = get_stockfish_eval_train(str(board_after.fen())) * color_factor
    eval_change = (eval_after - eval_before)  # Positive means improvement, negative means blunder

    # Major Game Outcomes
    final_reward = 0
    if game_result is not None:
        if game_result == 0:  # Draw
            final_reward = DRAW_PENALTY
        else:
            final_reward = game_result * OUTCOME_REWARD  # Win/Loss impact

    # Material Gain/Loss
    material_reward = 0
    pieces_before = board_before.piece_map()
    pieces_after = board_after.piece_map()
    
    for square, piece in pieces_before.items():
        if square not in pieces_after:  # Piece disappeared
            if piece.color != board_before.turn:  # Captured opponent's piece
                material_reward += PIECE_VALUES[piece.symbol().lower()] * MATERIAL_WEIGHT
            else:  # Lost own piece
                material_reward -= PIECE_VALUES[piece.symbol().lower()] * MATERIAL_WEIGHT

    # Pawn Promotion
    promotion_reward = PROMOTION_REWARD * color_factor if len(str(athena_move)) == 5 else 0

    # Passed Pawn Advancement
    passed_pawn_reward = 0
    moved_piece = board_after.piece_at(athena_move.to_square)
    if moved_piece and moved_piece.piece_type == chess.PAWN:
        passed_pawn_reward = PASSED_PAWN_REWARD * color_factor

    # Center Control Reward
    center_control_reward = CENTER_CONTROL_REWARD if str(athena_move)[-2:] in {"d4", "e4", "d5", "e5"} else 0

    # King Safety Penalty (if king is exposed in midgame)
    king_safety_penalty = 0
    if len(pieces_after) > 14:  # Midgame condition
        king_square = board_after.king(board_after.turn)
        rank, file = divmod(king_square, 8)
        if rank in {0, 7} or file in {0, 7}:  # King is on the edge
            king_safety_penalty = KING_SAFETY_PENALTY * color_factor

    # Avoid Repetition
    repetition_penalty = AVOID_REPETITION_PENALTY if board_after.is_repetition(3) else 0

    # Total Reward Computation
    total_reward = (
        final_reward + material_reward + eval_change +
        promotion_reward + passed_pawn_reward +
        center_control_reward + king_safety_penalty +
        repetition_penalty
    )

    return total_reward
