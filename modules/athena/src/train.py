import os
import sys
import chess
import numpy as np
import tensorflow as tf
from datetime import datetime
from typing import Dict, Any
from tqdm import tqdm
from tensorflow import keras
from tensorflow.keras.models import load_model
from pathlib import Path

# Set up root path for project-wide imports
ROOT_PATH = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_PATH))

# Local module imports
from modules.athena.src.aegis_net import AegisNet
from modules.athena.src.prometheus_net import PrometheusNet
from modules.athena.src.hybrid_trainer import HybridTrainer
from modules.athena.src.evaluate import evaluate_model

# Define paths for saving models and using Stockfish
MODELS_DIR = ROOT_PATH / 'models' / 'athena'
STOCKFISH_PATH = ROOT_PATH / "modules/shared/stockfish/stockfish-windows-x86-64-avx2.exe"

class Trainer:
    def __init__(self, weight_path="models/athena_hybrid_stock50self2.weights.h5"):
        # Initialize the neural network and training components
        # self.network = AegisNet()
        self.network = PrometheusNet()
        self.network([  # builds the model
            np.zeros((1, 8, 8, 20), dtype=np.float32),
            np.zeros((1, 50), dtype=np.float32),
            np.zeros((1, 8, 8), dtype=np.float32),
            np.zeros((1, 8, 8), dtype=np.float32)
        ])
        self.hybrid_trainer = HybridTrainer(self.network, str(STOCKFISH_PATH))
        self.weight_path = weight_path
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

    def train(self, num_iterations=1, initial_elo=0):
        print("Starting Hybrid AlphaZero training...")

        # Timestamp for model saving
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Set up learning rate schedule
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps=10000,
            decay_rate=0.9,
            staircase=True
        )

        # Compile the model with policy and value heads
        self.network.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
            loss={
                'policy': 'categorical_crossentropy',
                'promotion': 'categorical_crossentropy',
                'value': 'mean_squared_error'
            },
            loss_weights={
                'policy': 1.0,
                'promotion': 0.2,
                'value': 1.0
            },
            metrics={
                'policy': ['accuracy'],
                'promotion': ['accuracy'],
                'value': ['mae']
            }
        )

        # Optionally resume training from a full model
        # self.network.alpha_model = load_model("models/athena/full_athena_model_20240507_elo_1180")

        # Save initial model weights
        initial_save_path = MODELS_DIR / f"athena_hybrid_initial_{timestamp}.weights.h5"
        self.network.save_weights(str(initial_save_path))
        print(f"Saved initial model: {initial_save_path.name}")

        # Initialize ELO
        self.hybrid_trainer.update_elo(initial_elo)
        current_elo = initial_elo
        print("Initial ELO updated.")

        try:
            for iteration in tqdm(range(num_iterations), desc="Training iterations"):
                run_evaluation = (iteration % 5 == 0)  # Evaluate every 5 iterations

                print("Getting curriculum configuration...")
                curriculum = self.hybrid_trainer.get_curriculum_config()
                print("Curriculum received.")
                print(f"\nIteration {iteration + 1}/{num_iterations}")
                print(f"Phase: {curriculum['positions']} | Focus: {', '.join(curriculum['focus_areas'])}")

                # Perform one training iteration
                metrics = self.hybrid_trainer.train_iteration(iteration, run_evaluation=run_evaluation)

                print("\nTraining Metrics:")
                if metrics.get('stockfish'):
                    print("Stockfish training results:")
                    for key, value in metrics['stockfish'].items():
                        print(f"  {key}: {value}")

                # Optionally save model checkpoint
                # save_path = MODELS_DIR / f"athena_hybrid_iter_{iteration}_{timestamp}.weights.h5"
                # self.network.save_weights(str(save_path))
                # print(f"Saved model checkpoint: {save_path.name}")

                # Update ELO
                if 'elo_estimate' in metrics['stockfish']:
                    current_elo = metrics['stockfish']['elo_estimate']
                else:
                    current_elo += 15  # Increment fallback if no estimate
                self.hybrid_trainer.update_elo(current_elo)
                print(f"Updated ELO: {current_elo}")

            # Final evaluation
            print("\nTraining completed. Evaluating final model against Stockfish...")
            final_elo = 0  # Placeholder — replace with: evaluate_model(self.network)
            print(f"\nFinal Model Estimated ELO: {final_elo}")

            final_save_path = MODELS_DIR / f"athena_hybrid_final_{timestamp}_elo_{int(final_elo)}.keras"
            self.network.save(str(final_save_path))
            print(f"Saved Final Model: {final_save_path.name}")

        except KeyboardInterrupt:
            print("\nTraining interrupted by user. Saving current model...")
            final_save_path = MODELS_DIR / f"athena_hybrid_final_interrupted_{timestamp}.keras"
            self.network.save(str(final_save_path))
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
