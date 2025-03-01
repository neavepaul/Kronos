PIECE_VALUES = {"p": 0.05, "n": 0.15, "b": 0.15, "r": 0.25, "q": 0.5, "k": 0}

OUTCOME_REWARD_WIN = 1.0
OUTCOME_REWARD_LOSS = -1.0
OUTCOME_REWARD_DRAW = 0.0

THREAT_PENALTY = {
    "p": -0.02, "n": -0.05, "b": -0.05, "r": -0.1, "q": -0.2, "k": 0  # king threats don't matter this way
}

def compute_reward(board_before, board_after, game_result):
    outcome_reward = 0
    if game_result is not None:
        if game_result == 0:
            outcome_reward = OUTCOME_REWARD_DRAW
        elif game_result == 1:
            outcome_reward = OUTCOME_REWARD_WIN
        elif game_result == -1:
            outcome_reward = OUTCOME_REWARD_LOSS

    material_reward = compute_material_reward(board_before, board_after)
    threat_penalty = compute_threat_penalty(board_after)

    reward = outcome_reward + material_reward + threat_penalty
    return reward

def compute_material_reward(board_before, board_after):
    reward = 0
    for square, piece in board_before.piece_map().items():
        if square not in board_after.piece_map():
            value = PIECE_VALUES[piece.symbol().lower()]
            if piece.color != board_before.turn:
                reward += value
            else:
                reward -= value
    return reward

def compute_threat_penalty(board):
    penalty = 0
    for square, piece in board.piece_map().items():
        if piece.color == board.turn:
            attackers = board.attackers(not piece.color, square)
            if len(attackers) > 0:
                penalty += THREAT_PENALTY[piece.symbol().lower()]
    return penalty
