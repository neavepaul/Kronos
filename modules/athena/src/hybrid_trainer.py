import chess
import numpy as np
import logging
from typing import Dict, Any, Tuple
from pathlib import Path
from tqdm import tqdm

from modules.athena.src.selfplay import SelfPlayTrainer
from modules.athena.src.stockfish_trainer import StockfishTrainer

logger = logging.getLogger('athena.hybrid_trainer')

class HybridTrainer:
    def __init__(self, athena, stockfish_path: str = None):
        """Initialize hybrid trainer with both self-play and Stockfish components."""
        logger.info("Initializing HybridTrainer")
        self.athena = athena
        self.self_play_trainer = SelfPlayTrainer(athena)
        self.stockfish_trainer = StockfishTrainer(athena, stockfish_path)
        self.current_elo = 0  # Track estimated ELO
        
    def train_iteration(self, iteration: int, run_evaluation: bool = False) -> Dict[str, Any]:
        print(f"\n[HybridTrainer] Iteration {iteration} (ELO: {self.current_elo})")
        phase, stockfish_ratio = self._get_training_phase()
        print(f"[HybridTrainer] Phase: {phase} | Stockfish: {stockfish_ratio:.2f}")

        metrics = {'stockfish': {}, 'overall': {}}
        total_positions = 0

        # ONLY stockfish training
        if stockfish_ratio > 0:
            metrics['stockfish'] = self.stockfish_trainer.train_iteration(iteration, run_evaluation)

        metrics['overall'] = self._combine_metrics(metrics['stockfish'], {}, stockfish_ratio)
        return metrics



    def _get_training_phase(self) -> Tuple[str, float]:
        """Determine current training phase and Stockfish ratio."""
        if self.current_elo < 1800:
            # Phase 1: Heavy Stockfish training initially
            phase = "initial"
            stockfish_ratio = 1  # 100% Stockfish
        elif 1800 <= self.current_elo < 2200:
            # Phase 2: More balanced training
            phase = "transition"
            stockfish_ratio = 0.8  # 80% Stockfish, 20% self-play
        else:
            # Phase 3: Focus more on self-play for creative learning
            phase = "advanced"
            stockfish_ratio = 0.45  # 45% Stockfish, 55% self-play
        
        return phase, stockfish_ratio
    
    def _combine_metrics(
        self, 
        stockfish_metrics: Dict[str, float], 
        selfplay_metrics: Dict[str, float],
        stockfish_ratio: float
    ) -> Dict[str, float]:
        """Combine metrics from both training sources."""
        combined = {}
        
        # Common metrics between both sources
        common_metrics = set(stockfish_metrics.keys()) & set(selfplay_metrics.keys())
        for metric in common_metrics:
            # Weight the metrics based on the ratio
            combined[metric] = (
                stockfish_metrics[metric] * stockfish_ratio +
                selfplay_metrics[metric] * (1 - stockfish_ratio)
            )
        
        # Add unique metrics from each source
        for metric, value in stockfish_metrics.items():
            if metric not in combined:
                combined[f'stockfish_{metric}'] = value
                
        for metric, value in selfplay_metrics.items():
            if metric not in combined:
                combined[f'selfplay_{metric}'] = value
        
        return combined
    
    def _log_metrics(self, metrics: Dict[str, Dict[str, float]]):
        """Log training metrics in a readable format."""
        logger.info("\nTraining Metrics Summary:")
        
        if metrics['stockfish']:
            logger.info("\nStockfish Training:")
            for key, value in metrics['stockfish'].items():
                if isinstance(value, float):
                    logger.info(f"  {key}: {value:.4f}")
                else:
                    logger.info(f"  {key}: {value}")
        
        if metrics['self_play']:
            logger.info("\nSelf-play Training:")
            for key, value in metrics['self_play'].items():
                if isinstance(value, float):
                    logger.info(f"  {key}: {value:.4f}")
                else:
                    logger.info(f"  {key}: {value}")
        
        logger.info("\nOverall Metrics:")
        for key, value in metrics['overall'].items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")
    
    def update_elo(self, elo: float):
        """Update current ELO estimate."""
        self.current_elo = elo
        logger.info(f"Updated ELO rating to {elo}")
        
    def get_curriculum_config(self) -> Dict[str, Any]:
        """Get curriculum learning configuration based on current phase."""
        phase, _ = self._get_training_phase()
        
        if phase == "initial":
            return {
                'positions': 'simple',
                'stockfish_depth': 10,
                'focus_areas': ['basic_tactics', 'piece_coordination']
            }
        elif phase == "transition":
            return {
                'positions': 'moderate',
                'stockfish_depth': 15,
                'focus_areas': ['positional_play', 'pawn_structure']
            }
        else:  # advanced
            return {
                'positions': 'complex',
                'stockfish_depth': 20,
                'focus_areas': ['strategic_planning', 'endgame_technique']
            }