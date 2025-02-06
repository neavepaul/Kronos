import tensorflow as tf
import os
from model import get_model
from data_generator import HDF5DataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import datetime


current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# Training parameters
BATCH_SIZE = 64
EPOCHS = 15
MODEL_SAVE_PATH = f"models/athena_trained_{current_time}.h5"
LOG_DIR = "logs/athena"

# Ensure necessary directories exist
os.makedirs("models", exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Load Athena model
model = get_model()

# Extract move vocabulary layer from model (ensures consistency)
move_vocab_layer = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.StringLookup)][0]

# Initialize Data Generator
train_generator = HDF5DataGenerator(batch_size=BATCH_SIZE, move_vocab=move_vocab_layer)

# TensorBoard callback
# tensorboard_callback = TensorBoard(
#     log_dir=LOG_DIR,  
#     histogram_freq=1,  
#     write_graph=True,  
#     write_images=True,  
#     update_freq="epoch"  
# )

# Model Checkpointing (Save best model)
checkpoint_callback = ModelCheckpoint(
    filepath=MODEL_SAVE_PATH,
    save_best_only=True,
    monitor="loss",  # Track total loss instead of move_probs_loss
    mode="min",
    verbose=1
)

# Train the model with callbacks
model.fit(
    train_generator,
    epochs=EPOCHS,
    callbacks=[checkpoint_callback]
)
# model.fit(
#     train_generator,
#     epochs=EPOCHS,
#     callbacks=[tensorboard_callback, checkpoint_callback]
# )

# Save the final trained model
model.save(MODEL_SAVE_PATH)
print(f"Model saved at {MODEL_SAVE_PATH}")
