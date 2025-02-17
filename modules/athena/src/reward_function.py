import numpy as np
import chess

# Standard Piece Values
PIECE_VALUES = {"p": 1, "n": 3, "b": 3, "r": 5, "q": 9, "k": 0}  # King has no material value

# Reward Scaling Factors with Justifications so I don't doubt them later LOL :)
MATERIAL_WEIGHT = 0.1  # Material shaping should not overpower the final win/loss reward
OUTCOME_REWARD = 1.0    # Winning should be the highest priority
DRAW_PENALTY = -0.5     # Penalize stalemates to avoid Athena trading everything into draws
CASTLING_REWARD = 0.3   # Encourage castling only if performed
PROMOTION_REWARD = 5.0  # High reward for pawn promotion
EN_PASSANT_REWARD = 1.5 # Slightly higher than a normal pawn capture
FORK_REWARD = 0.5       # Reward for attacking multiple opponent pieces
PIN_REWARD = 0.4        # Reward for pinning opponent’s piece
SKEWER_REWARD = 0.6     # Reward for skewering an opponent’s high-value piece
TRAPPED_PIECE_REWARD = 0.5  # Reward for limiting opponent piece movement
PASSED_PAWN_ADVANCE_REWARD = 0.5  # Encourages pawn promotion strategy
CHECK_REWARD = 0.2      # Encourages offensive play
KING_ACTIVITY_REWARD = 0.3  # Encourages king activation in endgame
AVOID_REPETITION_PENALTY = -0.3  # Penalize unnecessary repetition

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

    # Determine Athena's color (who is playing this turn)
    color_factor = 1 if board_before.turn == chess.WHITE else -1  # White → +1, Black → -1

    # Track Material Changes (Piece Captures)
    pieces_before = board_before.piece_map()
    pieces_after = board_after.piece_map()

    captured_piece_value = 0
    lost_piece_value = 0

    # Detect captured pieces
    for square, piece in pieces_before.items():
        if square not in pieces_after:  # Piece disappeared from board
            if piece.color != board_before.turn:  # Opponent’s piece → We captured it
                captured_piece_value += PIECE_VALUES[piece.symbol().lower()]
            else:  # Our piece → We lost it
                lost_piece_value += PIECE_VALUES[piece.symbol().lower()]

    # Compute Material Reward
    material_reward = (captured_piece_value - lost_piece_value) * color_factor * MATERIAL_WEIGHT

    # Reward for **Correct** Castling (Only if performed)
    castling_reward = 0
    if board_before.has_castling_rights(board_before.turn) and not board_after.has_castling_rights(board_before.turn):
        if str(athena_move) in ["e1g1", "e1c1", "e8g8", "e8c8"]:  # Kingside or Queenside castling
            castling_reward = CASTLING_REWARD * color_factor

    # Reward Pawn Promotion
    promotion_reward = 0
    if len(str(athena_move)) == 5:  # e.g., "e7e8q" (promotion move has 5 chars)
        promotion_reward = PROMOTION_REWARD * color_factor

    # Reward En Passant
    en_passant_reward = 0
    if board_before.is_en_passant(chess.Move.from_uci(str(athena_move))):
        en_passant_reward = EN_PASSANT_REWARD * color_factor

    # Reward for Giving Check
    check_reward = 0
    if board_after.is_check():
        check_reward = CHECK_REWARD * color_factor

    # Reward for King Activity in Endgame
    king_activity_reward = 0
    if len(pieces_after) <= 10:  # Define endgame as 10 or fewer pieces
        king_square = board_after.king(board_after.turn)
        rank, file = divmod(king_square, 8)
        if 2 <= rank <= 5 and 2 <= file <= 5:  # Encourage king to move towards center
            king_activity_reward = KING_ACTIVITY_REWARD * color_factor

    # Reward Passed Pawn Advancement
    passed_pawn_reward = 0
    moved_square = athena_move.to_square  # The square where the piece moved
    moved_piece = board_after.piece_at(moved_square)

    if moved_piece and moved_piece.piece_type == chess.PAWN:
        file, rank = chess.square_file(moved_square), chess.square_rank(moved_square)  

        # Check if it’s a passed pawn (no opposing pawns blocking it)
        if not any(
            board_after.piece_at(chess.square(adj_file, r))
            for r in range(8)
            for adj_file in [file - 1, file + 1]  # Check adjacent files
            if 0 <= adj_file <= 7
        ):
            passed_pawn_reward = PASSED_PAWN_ADVANCE_REWARD * color_factor

    # Avoid Threefold Repetition (Discourage repeating moves when winning)
    repetition_penalty = 0
    if board_after.is_repetition(3):  # If the same position has repeated three times
        repetition_penalty = AVOID_REPETITION_PENALTY * color_factor

    # Check for Forks, Pins, and Skewers
    fork_reward = 0
    pin_reward = 0
    skewer_reward = 0
    for move in board_after.legal_moves:
        board_after.push(move)
        attacked_squares = {m.to_square for m in board_after.legal_moves}
        board_after.pop()

        # Fork Detection (Two pieces attacked at once)
        attacked_pieces = [square for square in attacked_squares if board_after.piece_at(square)]
        if len(attacked_pieces) >= 2:
            fork_reward += FORK_REWARD * color_factor

        # Pin Detection
        if board_after.is_pinned(not board_after.turn, move.from_square):
            pin_reward += PIN_REWARD * color_factor

        # Skewer Detection (Higher-value piece forced to move)
        for attacked in attacked_pieces:
            if board_after.piece_at(attacked) and board_after.piece_at(attacked).piece_type in [chess.QUEEN, chess.ROOK]:
                skewer_reward += SKEWER_REWARD * color_factor

    # Reward Trapping Opponent’s Piece (If it’s immobilized)
    trapped_piece_reward = 0
    for square, piece in pieces_after.items():
        if piece.color != board_after.turn:  # Opponent's piece
            if not any(board_after.is_legal(m) for m in board_after.generate_legal_moves()):
                trapped_piece_reward += TRAPPED_PIECE_REWARD * color_factor

    # Final Game Result Reward
    final_reward = 0
    if game_result is not None:
        if game_result == 0:  # Draw (Stalemate or repetition)
            final_reward = DRAW_PENALTY
        else:
            final_reward = game_result * OUTCOME_REWARD * color_factor

    # Total Reward
    total_reward = (
        material_reward + castling_reward + promotion_reward + en_passant_reward + check_reward +
        fork_reward + pin_reward + skewer_reward + trapped_piece_reward + 
        passed_pawn_reward + king_activity_reward + repetition_penalty + final_reward
    )

    return total_reward
