import os
import tensorflow as tf
from datetime import datetime
from model import get_model
from data_generator import HDF5DataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.0001
NUM_TRANSFORMERS = 2
EMBED_DIM = 128
DROPOUT_RATE = 0.3

# Model Save Path
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
MODEL_SAVE_PATH = f"models/athena_gm_trained_{timestamp}_{EPOCHS}epochs.keras"
os.makedirs("models", exist_ok=True)

# Log File Path
LOG_FILE = f"training_log_{timestamp}.txt"

# Custom Callback for Logging Losses & Batch Information
class LossLoggerCallback(Callback):
    def on_train_begin(self, logs=None):
        """Initialize log file"""
        with open(LOG_FILE, "w") as f:
            f.write("Batch\tMove_Loss\tCriticality_Loss\tSamples_Per_Batch\n")

    def on_batch_end(self, batch, logs=None):
        """Log loss values at the end of each batch"""
        move_loss = logs.get("move_output_loss", 0)
        criticality_loss = logs.get("criticality_loss", 0)
        samples_per_batch = BATCH_SIZE  # Since batch size is fixed

        with open(LOG_FILE, "a") as f:
            f.write(f"{batch}\t{move_loss:.6f}\t{criticality_loss:.6f}\t{samples_per_batch}\n")

# Load Data
train_generator = HDF5DataGenerator(batch_size=BATCH_SIZE)

# Load Model with dynamic hyperparameters
model = get_model(
    num_transformers=NUM_TRANSFORMERS,
    embed_dim=EMBED_DIM,
    dropout_rate=DROPOUT_RATE,
    learning_rate=LEARNING_RATE
)

# Callbacks
checkpoint_callback = ModelCheckpoint(
    "models/best_checkpoint.keras",
    monitor="loss",
    save_best_only=True
)

early_stopping = EarlyStopping(
    monitor="loss",
    patience=3,
    restore_best_weights=True
)

# Add Loss Logging Callback
loss_logger = LossLoggerCallback()

# Train Model
model.fit(train_generator, epochs=EPOCHS, callbacks=[checkpoint_callback, early_stopping, loss_logger])

# Save Final Model
model.save(MODEL_SAVE_PATH)

print(f"Model saved at: {MODEL_SAVE_PATH}")
print(f"Training Complete! Loss values logged in {LOG_FILE}")
