# train_rl.py
import os
import chess
import chess.engine
import numpy as np
import tensorflow as tf
from model import get_model
from q_network import QNetwork
from actor_critic import ActorCritic
from data_generator import HDF5DataGenerator

STOCKFISH_PATH = "stockfish/stockfish-windows-x86-64-avx2.exe"
BATCH_SIZE = 64
EPOCHS = 20
MODEL_SAVE_PATH = "models/athena_rl_trained.h5"

os.makedirs("models", exist_ok=True)

# Load Athena (Pretrained on GM Data)
athena = get_model()
athena.load_weights("models/athena_gm_trained.h5")

# Load Q-Network & Actor-Critic
q_network = QNetwork()
actor_critic = ActorCritic()

# Define Optimizers
q_optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
ac_optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)

def evaluate_board(board, engine):
    """Returns Stockfish evaluation score from White's perspective."""
    result = engine.analyse(board, chess.engine.Limit(depth=10))
    return result["score"].relative.score(mate_score=10000) if result["score"].relative else 0  

def rl_loss(y_true, y_pred, white_to_move):
    delta_score = y_true - y_pred
    return tf.reduce_mean(tf.where(white_to_move, -delta_score, delta_score))

def train_rl():
    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
        train_generator = HDF5DataGenerator(batch_size=BATCH_SIZE)

        for epoch in range(EPOCHS):
            print(f"\n🚀 RL Training Epoch {epoch + 1}/{EPOCHS}...\n")
            total_loss = 0

            for batch in train_generator:
                inputs, _ = batch
                fens, move_histories, legal_moves_mask, eval_scores = inputs.values()
                batch_size = fens.shape[0]
                batch_loss = 0

                for i in range(batch_size):
                    board = chess.Board()
                    board.set_fen(fens[i])

                    # ✅ **Athena Predicts a Move**
                    pred_move_pffttu, criticality = athena.predict({
                        "fen_input": fens[i:i+1],
                        "move_seq": move_histories[i:i+1],
                        "legal_mask": legal_moves_mask[i:i+1],
                        "eval_score": eval_scores[i:i+1]
                    })

                    # Convert PFFTTU Move Format → Chess Move
                    try:
                        piece, from_square, to_square, _, _, upgrade = pred_move_pffttu[0]
                        move = chess.Move(from_square, to_square, promotion=upgrade)
                        if not board.is_legal(move):
                            continue
                    except:
                        continue

                    board.push(move)

                    # ✅ **Q-Network Move Evaluation**
                    q_values = q_network(fens[i:i+1])
                    best_move_index = np.argmax(q_values.numpy())
                    chosen_q_value = q_values.numpy()[0, best_move_index]

                    # ✅ **Evaluate Board Before & After Move**
                    eval_before = evaluate_board(board, engine)
                    eval_after = evaluate_board(board, engine)
                    delta_score = eval_after - eval_before

                    # ✅ **Compute Reward**
                    white_to_move = board.turn == chess.WHITE
                    reward = -delta_score if white_to_move else delta_score  # Reward = Positive for Good Move

                    # ✅ **Q-Network Loss Update**
                    with tf.GradientTape() as tape:
                        pred_q_value = q_network(fens[i:i+1])[0, best_move_index]
                        loss_q = tf.keras.losses.MSE(reward, pred_q_value)
                    gradients_q = tape.gradient(loss_q, q_network.trainable_variables)
                    q_optimizer.apply_gradients(zip(gradients_q, q_network.trainable_variables))

                    # ✅ **Actor-Critic Update**
                    with tf.GradientTape() as tape:
                        policy_probs, state_value = actor_critic(fens[i:i+1])
                        advantage = reward - state_value  # Advantage function A(s, a) = Q(s,a) - V(s)
                        log_policy = tf.math.log(policy_probs[0, best_move_index])
                        loss_actor = -log_policy * advantage  # Policy Gradient Loss
                        loss_critic = tf.keras.losses.MSE(reward, state_value)
                        loss_ac = loss_actor + loss_critic
                    gradients_ac = tape.gradient(loss_ac, actor_critic.trainable_variables)
                    ac_optimizer.apply_gradients(zip(gradients_ac, actor_critic.trainable_variables))

                    batch_loss += loss_ac.numpy()

                total_loss += batch_loss
                print(f"📉 Batch Loss: {batch_loss:.4f}")

            print(f"📉 Epoch {epoch + 1} Loss: {total_loss:.4f}")

            # ✅ Save Model
            athena.save(MODEL_SAVE_PATH)
            print(f"✅ Model saved at {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_rl()
