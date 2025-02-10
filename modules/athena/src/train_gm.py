import os
import tensorflow as tf
from datetime import datetime
from model import get_model
from data_generator import HDF5DataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

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

# Load Data
train_generator = HDF5DataGenerator(batch_size=BATCH_SIZE)

# Load Model with dynamic hyperparameters
model = get_model(
    num_transformers=NUM_TRANSFORMERS,
    embed_dim=EMBED_DIM,
    dropout_rate=DROPOUT_RATE,
    learning_rate=LEARNING_RATE
)

# ✅ Fixed ModelCheckpoint (No 'save_format' argument)
checkpoint_callback = ModelCheckpoint(
    "models/best_athena.keras",  # Saves the best model automatically
    monitor="loss",
    save_best_only=True
)

early_stopping = EarlyStopping(
    monitor="loss",
    patience=3,
    restore_best_weights=True
)

# Train Model
model.fit(train_generator, epochs=EPOCHS, callbacks=[checkpoint_callback, early_stopping])

# ✅ Save Final Model
model.save(MODEL_SAVE_PATH)

print(f"Model saved at: {MODEL_SAVE_PATH}")
print("Training Complete!")