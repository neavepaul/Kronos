import numpy as np

def compute_reward(stockfish_eval_before, stockfish_eval_after, athena_move, stockfish_move, game_result):
    """
    Computes the reward for Athena based on Stockfish's evaluation.
    
    Parameters:
        stockfish_eval_before (float): Stockfish evaluation before Athena's move.
        stockfish_eval_after (float): Stockfish evaluation after Athena's move.
        athena_move (str): The move chosen by Athena in UCI format.
        stockfish_move (str): The move chosen by Stockfish in UCI format.
        game_result (int): The final game result (+1 for win, -1 for loss, 0 for draw).
    
    Returns:
        float: Computed reward for reinforcement learning.
    """
    
    # Normalize evaluations (convert centipawns to pawn units)
    eval_change = (stockfish_eval_after - stockfish_eval_before) / 100.0
    
    # Check if Athena's move matches Stockfish's best move
    move_match_bonus = 1.0 if athena_move == stockfish_move else -0.5
    
    # Final game outcome bonus (big incentive for winning, penalty for losing)
    game_result_bonus = 10.0 * game_result
    
    # Compute total reward
    reward = eval_change + move_match_bonus + game_result_bonus
    
    return reward

# Example Usage
if __name__ == "__main__":
    stockfish_eval_before = 50  # Stockfish evaluation before Athena's move (50 centipawns)
    stockfish_eval_after = 70   # Stockfish evaluation after Athena's move (70 centipawns)
    athena_move = "e2e4"
    stockfish_move = "e2e4"  # Best move chosen by Stockfish
    game_result = 1  # Win (+1), Loss (-1), Draw (0)

    reward = compute_reward(stockfish_eval_before, stockfish_eval_after, athena_move, stockfish_move, game_result)
    print(f"Computed Reward: {reward}")
