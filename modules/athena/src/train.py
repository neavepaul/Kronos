import os
import sys
import chess
import numpy as np
import tensorflow as tf
from datetime import datetime
from typing import Dict, Any
from tqdm import tqdm

from pathlib import Path
ROOT_PATH = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_PATH))

from modules.athena.src.alpha_net import AlphaNet
from modules.athena.src.hybrid_trainer import HybridTrainer

MODELS_DIR = Path(__file__).resolve().parents[3] / 'models' / 'athena'
STOCKFISH_PATH = ROOT_PATH / "modules/shared/stockfish/stockfish-windows-x86-64-avx2.exe"

class Trainer:
    def __init__(self):
        self.network = AlphaNet()
        self.hybrid_trainer = HybridTrainer(self.network, str(STOCKFISH_PATH))
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

    def train(self, num_iterations=1, initial_elo=0):
        print("Starting Hybrid AlphaZero training...")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        initial_save_path = MODELS_DIR / f"athena_hybrid_{timestamp}.weights.h5"
        self.network.alpha_model.save_weights(str(initial_save_path))
        print(f"Saved initial model: {initial_save_path.name}")

        self.hybrid_trainer.update_elo(initial_elo)
        current_elo = initial_elo
        print("Updated ELO")

        try:
            for iteration in tqdm(range(num_iterations), desc="Training iterations"):
                run_evaluation = (iteration % 5 == 0)  # Evaluate every 5 iterations
                print("Getting curriculum configuration...")
                curriculum = self.hybrid_trainer.get_curriculum_config()
                print("Received curriculum configuration")
                print(f"\nIteration {iteration + 1}/{num_iterations}")
                print(f"Phase: {curriculum['positions']} | Focus: {', '.join(curriculum['focus_areas'])}")

                metrics = self.hybrid_trainer.train_iteration(iteration, run_evaluation=run_evaluation)

                print("\nTraining Metrics:")
                if metrics.get('stockfish'):
                    print("Stockfish training:")
                    for key, value in metrics['stockfish'].items():
                        print(f"  {key}: {value}")

                save_path = MODELS_DIR / f"athena_hybrid_iter_{iteration}_{timestamp}.weights.h5"
                self.network.alpha_model.save_weights(str(save_path))
                print(f"Saved model checkpoint: {save_path.name}")

                if 'elo_estimate' in metrics['stockfish']:
                    current_elo = metrics['stockfish']['elo_estimate']
                else:
                    current_elo += 15

                self.hybrid_trainer.update_elo(current_elo)
                print(f"Updated ELO: {current_elo}")

            print("\n🎯 Training finished. Evaluating final model against Stockfish...")

            from modules.athena.src.evaluate import evaluate_model
            final_elo = evaluate_model(self.network)
            print(f"\n🏆 Final Model Estimated ELO: {final_elo}")

            final_save_path = MODELS_DIR / f"athena_hybrid_final_{timestamp}_elo_{int(final_elo)}.weights.h5"
            self.network.alpha_model.save_weights(str(final_save_path))
            print(f"Saved Final Model: {final_save_path.name}")

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
        num_iterations=50,
        initial_elo=0
    )

if __name__ == "__main__":
    main()
