import chess
import chess.engine
import logging
import random
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict

logger = logging.getLogger('athena.evaluate')

class EloEvaluator:
    def __init__(self, stockfish_path: Optional[str] = None):
        if stockfish_path is None:
            stockfish_path = str(Path(__file__).parent.parent.parent / 'shared' / 'stockfish' / 'stockfish-windows-x86-64-avx2.exe')
            
        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        # Configure different Stockfish skill levels for ELO estimation
        self.elo_levels = {
            0: {'Skill Level': 0, 'ELO': 1100},
            5: {'Skill Level': 5, 'ELO': 1500},
            10: {'Skill Level': 10, 'ELO': 1800},
            15: {'Skill Level': 15, 'ELO': 2100},
            20: {'Skill Level': 20, 'ELO': 2400}
        }
        
    def play_match(self, athena, skill_level: int, num_games: int = 10) -> Dict:
        """Play a match against Stockfish at given skill level."""
        self.engine.configure({'Skill Level': skill_level})
        wins = draws = losses = 0
        
        for game_id in range(num_games):
            board = chess.Board()
            # Alternate colors
            athena_white = game_id % 2 == 0
            
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
        """Estimate ELO rating through matches against different Stockfish levels."""
        logger.info("Starting ELO estimation")
        results = {}
        
        # Start with middle level
        current_level = 10
        score = self.play_match(athena, current_level)['score']
        
        if score > 0.6:  # Doing well, try higher levels
            levels_to_test = [15, 20]
        elif score < 0.4:  # Struggling, try lower levels
            levels_to_test = [5, 0]
        else:  # Roughly equal, test adjacent levels
            levels_to_test = [5, 15]
            
        results[current_level] = score
        
        # Test selected levels
        for level in levels_to_test:
            results[level] = self.play_match(athena, level)['score']
            
        # Estimate ELO through interpolation
        elo_estimate = self._interpolate_elo(results)
        logger.info(f"Estimated ELO: {elo_estimate}")
        
        return elo_estimate
    
    def _interpolate_elo(self, results: Dict[int, float]) -> float:
        """Interpolate ELO rating from match results."""
        # Find closest skill levels
        scores = [(level, score) for level, score in results.items()]
        scores.sort(key=lambda x: x[1], reverse=True)
        
        if len(scores) == 1:
            return self.elo_levels[scores[0][0]]['ELO']
            
        # Linear interpolation between closest levels
        for i in range(len(scores) - 1):
            level1, score1 = scores[i]
            level2, score2 = scores[i + 1]
            
            if score1 >= 0.5 >= score2:
                elo1 = self.elo_levels[level1]['ELO']
                elo2 = self.elo_levels[level2]['ELO']
                # Interpolate based on score difference
                ratio = (0.5 - score2) / (score1 - score2)
                return elo2 + ratio * (elo1 - elo2)
        
        # If no interpolation possible, return closest match
        return self.elo_levels[scores[0][0]]['ELO']
    
    def __del__(self):
        if hasattr(self, 'engine'):
            self.engine.quit()

def evaluate_model(athena) -> float:
    """Evaluate a model and return its estimated ELO rating."""
    evaluator = EloEvaluator()
    return evaluator.estimate_elo(athena)
