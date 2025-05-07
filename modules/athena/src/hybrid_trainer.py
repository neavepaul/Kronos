import os, sys
import chess
import numpy as np
import logging
import random
from typing import Dict, Any, Tuple
from pathlib import Path
from tqdm import tqdm

# Dynamically set the root path to the "Kronos" directory
ROOT_PATH = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_PATH))
from modules.athena.src.selfplay import SelfPlayTrainer
from modules.athena.src.stockfish_trainer import StockfishTrainer
from modules.athena.src.dual_stockfish_trainer import StockfishDualTrainer

logger = logging.getLogger('athena.hybrid_trainer')

class HybridTrainer:
    def __init__(self, athena, stockfish_path: str = None):
        logger.info("Initializing HybridTrainer")
        self.athena = athena
        self.self_play_trainer = SelfPlayTrainer(athena)
        self.stockfish_trainer = StockfishTrainer(athena, stockfish_path)
        self.dual_stockfish_trainer = StockfishDualTrainer(athena, stockfish_path)
        self.current_iteration = 0

    def train_iteration(self, iteration: int, run_evaluation: bool = False) -> Dict[str, Any]:
        logger.info(f"\n[HybridTrainer] Iteration {iteration}")
        phase, stockfish_ratio = self._get_training_phase()
        logger.info(f"[HybridTrainer] Phase: {phase} | Stockfish Ratio: {stockfish_ratio:.2f}")

        metrics = {'stockfish': {}, 'selfplay': {}, 'overall': {}}

        if phase == "initial":
            metrics['stockfish'] = self.dual_stockfish_trainer.train_from_stronger_stockfish()
        else:
            use_stockfish = random.random() < stockfish_ratio
            if use_stockfish:
                metrics['stockfish'] = self.stockfish_trainer.train_iteration(iteration, run_evaluation)
            else:
                metrics['selfplay'] = self.self_play_trainer.train_iteration(iteration)

        metrics['overall'] = self._combine_metrics(metrics['stockfish'], metrics['selfplay'], stockfish_ratio)
        self._log_metrics(metrics)
        self.current_iteration += 1
        return metrics

    def _get_training_phase(self) -> Tuple[str, float]:
        """Determine training phase and stockfish/selfplay ratio."""
        if self.current_iteration < 100:
            phase = "initial"
            stockfish_ratio = 1.0  # 100% Dual Stockfish
        elif 100 <= self.current_iteration < 200:
            phase = "transition"
            stockfish_ratio = 0.7  # 70% Stockfish, 30% Self-Play
        else:
            phase = "advanced"
            stockfish_ratio = 0.4  # 40% Stockfish, 60% Self-Play
        return phase, stockfish_ratio

    def _combine_metrics(
        self,
        stockfish_metrics: Dict[str, float],
        selfplay_metrics: Dict[str, float],
        stockfish_ratio: float
    ) -> Dict[str, float]:
        """Combine metrics from both training sources."""
        combined = {}

        if stockfish_metrics and selfplay_metrics:
            for key in set(stockfish_metrics.keys()) & set(selfplay_metrics.keys()):
                combined[key] = (
                    stockfish_metrics[key] * stockfish_ratio +
                    selfplay_metrics[key] * (1 - stockfish_ratio)
                )
        elif stockfish_metrics:
            combined.update(stockfish_metrics)
        elif selfplay_metrics:
            combined.update(selfplay_metrics)

        return combined

    def _log_metrics(self, metrics: Dict[str, Dict[str, float]]):
        """Log training metrics in a readable format."""
        logger.info("\nTraining Metrics Summary:")

        if metrics['stockfish']:
            logger.info("\nStockfish Training:")
            for key, value in metrics['stockfish'].items():
                logger.info(f"  {key}: {value:.4f}")

        if metrics['selfplay']:
            logger.info("\nSelf-Play Training:")
            for key, value in metrics['selfplay'].items():
                logger.info(f"  {key}: {value:.4f}")

        logger.info("\nOverall Metrics:")
        for key, value in metrics['overall'].items():
            logger.info(f"  {key}: {value:.4f}")

    def update_elo(self, elo: float):
        pass  # Deprecated for now, can rewire to real evaluation if needed

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
