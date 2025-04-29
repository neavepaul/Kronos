import chess
import chess.engine
import logging
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict

logger = logging.getLogger('athena.evaluate')

class EloEvaluator:
    def __init__(self, stockfish_path: Optional[str] = None):
        if stockfish_path is None:
            stockfish_path = str(Path(__file__).parent.parent.parent / 'shared' / 'stockfish' / 'stockfish-windows-x86-64-avx2.exe')

        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        self.skill_levels = [5, 7, 9, 11, 13, 15]  # Fine-grained ELO test levels

    def play_match(self, athena, skill_level: int, num_games: int = 12) -> Dict:
        """Play a match against Stockfish at given skill level."""
        self.engine.configure({'Skill Level': skill_level})
        wins = draws = losses = 0

        for game_id in range(num_games):
            board = chess.Board()
            athena_white = game_id % 2 == 0  # Alternate colors

            while not board.is_game_over():
                if board.turn == chess.WHITE:
                    if athena_white:
                        move_probs, _ = athena.predict(board)
                        move = max(move_probs, key=move_probs.get)
                    else:
                        move = self.engine.play(board, chess.engine.Limit(time=0.1)).move
                else:
                    if athena_white:
                        move = self.engine.play(board, chess.engine.Limit(time=0.1)).move
                    else:
                        move_probs, _ = athena.predict(board)
                        move = max(move_probs, key=move_probs.get)
                board.push(move)

            result = board.result()
            if result == '1-0':
                wins += 1 if athena_white else 0
                losses += 0 if athena_white else 1
            elif result == '0-1':
                wins += 0 if athena_white else 1
                losses += 1 if athena_white else 0
            else:
                draws += 1

        return {
            'wins': wins,
            'draws': draws,
            'losses': losses,
            'score': (wins + 0.5 * draws) / num_games
        }

    def estimate_elo(self, athena) -> float:
        """Estimate ELO rating through matches against multiple Stockfish levels."""
        logger.info("Starting ELO estimation")
        scores = {}

        for level in self.skill_levels:
            result = self.play_match(athena, level)
            scores[level] = result['score']
            logger.info(f"Skill {level}: {result}")

        # Calculate ELO against each Stockfish level
        elos = []
        for level, score in scores.items():
            stockfish_elo = self._stockfish_level_to_elo(level)
            estimated_elo = stockfish_elo + 400 * (score - 0.5)
            elos.append(estimated_elo)

        final_elo = np.mean(elos)
        logger.info(f"Estimated ELO: {final_elo:.2f}")

        return final_elo

    def _stockfish_level_to_elo(self, level: int) -> int:
        """Reddit-based realistic ELO mapping for Stockfish skill levels."""
        refined_mapping = {
            1: 1385, 2: 1460, 3: 1540, 4: 1620, 5: 1700,
            6: 1780, 7: 1860, 8: 1940, 9: 2020, 10: 2100,
            11: 2180, 12: 2260, 13: 2340, 14: 2420, 15: 2500,
            16: 2580, 17: 2660, 18: 2740, 19: 2820, 20: 2900
        }
        return refined_mapping.get(level, 2100)

    def __del__(self):
        if hasattr(self, 'engine'):
            self.engine.quit()


def evaluate_model(athena) -> float:
    """Evaluate a model and return its estimated ELO rating."""
    evaluator = EloEvaluator()
    return evaluator.estimate_elo(athena)


def quick_evaluate_model(athena) -> float:
    """Quick ELO evaluation (lightweight for training)."""
    evaluator = EloEvaluator()
    evaluator.skill_levels = [10]
    result = evaluator.play_match(athena, skill_level=10, num_games=6)
    stockfish_elo = evaluator._stockfish_level_to_elo(10)
    estimated_elo = stockfish_elo + 400 * (result['score'] - 0.5)
    return estimated_elo
