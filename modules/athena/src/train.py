import os
import sys
import chess
import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
from tqdm import tqdm

# Dynamically set the root path to the "Kronos" directory
ROOT_PATH = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_PATH))

from modules.athena.src.alpha_net import AlphaNet
from modules.athena.src.hybrid_trainer import HybridTrainer

MODELS_DIR = Path(__file__).resolve().parents[3] / 'models' / 'athena'
STOCKFISH_PATH = ROOT_PATH / "modules/shared/stockfish/stockfish-windows-x86-64-avx2.exe"

class Trainer:
    """Main training coordinator using hybrid approach."""
    
    def __init__(self):
        self.network = AlphaNet()
        self.hybrid_trainer = HybridTrainer(self.network, str(STOCKFISH_PATH))
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        
    def train(self, num_iterations=1, initial_elo=0):
        """Main training loop using hybrid approach."""
        print("Starting Hybrid AlphaZero training...")
        
        # Save initial model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        initial_save_path = MODELS_DIR / f"athena_hybrid_{timestamp}.weights.h5"
        self.network.alpha_model.save_weights(str(initial_save_path))
        print(f"Saved initial model: {initial_save_path.name}")
        
        # Set initial ELO rating
        self.hybrid_trainer.update_elo(initial_elo)
        current_elo = initial_elo
        
        try:
            for iteration in tqdm(range(num_iterations), desc="Training iterations"):
                # Get curriculum config for this phase
                curriculum = self.hybrid_trainer.get_curriculum_config()
                print(f"\nIteration {iteration + 1}/{num_iterations}")
                print(f"Current curriculum phase: {curriculum['positions']}")
                print(f"Focus areas: {', '.join(curriculum['focus_areas'])}")
                
                # Run hybrid training iteration
                metrics = self.hybrid_trainer.train_iteration(iteration)
                
                # Print metrics
                print("\nTraining Metrics:")
                if metrics.get('stockfish'):
                    print("Stockfish training:")
                    for key, value in metrics['stockfish'].items():
                        print(f"  {key}: {value}")
                if metrics.get('self_play'):
                    print("Self-play training:")
                    for key, value in metrics['self_play'].items():
                        print(f"  {key}: {value}")
                
                # Save model checkpoint
                save_path = MODELS_DIR / f"athena_hybrid_iter_{iteration}_{timestamp}.weights.h5"
                self.network.alpha_model.save_weights(str(save_path))
                print(f"Saved model checkpoint: {save_path.name}")
                
                # Update ELO estimate based on performance
                # Start conservative, increase faster with good performance
                if metrics.get('stockfish', {}).get('win_rate', 0) > 0.55:
                    elo_gain = 25  # Bigger jump if winning against Stockfish
                else:
                    elo_gain = 15  # Smaller improvement otherwise
                
                current_elo += elo_gain
                self.hybrid_trainer.update_elo(current_elo)
                print(f"Updated ELO estimate: {current_elo}")
        
        except KeyboardInterrupt:
            print("\nTraining interrupted. Saving final model...")
            final_save_path = MODELS_DIR / f"athena_hybrid_final_interrupted_{timestamp}.weights.h5"
            self.network.alpha_model.save_weights(str(final_save_path))
            print(f"Saved interrupted model: {final_save_path.name}")
        
        except Exception as e:
            print(f"Training error: {str(e)}")
            raise

def main():
    trainer = Trainer()
    trainer.train(
        num_iterations=1,  # Adjust based on training budget
        initial_elo=0    # Starting ELO rating
    )

if __name__ == "__main__":
    main()