import tensorflow as tf
import os
from model import get_model
from data_generator import HDF5DataGenerator

# Training parameters
BATCH_SIZE = 32
EPOCHS = 10
MODEL_SAVE_PATH = "models/athena_trained.h5"
LOG_DIR = "logs/athena"

# Ensure log directory exists
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# Initialize model
model = get_model()

# Initialize Data Generator
train_generator = HDF5DataGenerator(batch_size=BATCH_SIZE)

# TensorBoard callback
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=LOG_DIR,  # Directory where logs will be stored
    histogram_freq=1,  # Log weight histograms every epoch
    write_graph=True,  # Log computation graph
    write_images=True,  # Log model weights as images
    update_freq="epoch"  # Log at each epoch
)


# Start tensorboard server with: tensorboard --logdir=logs/
# View at localhost:6006

# Train the model with TensorBoard callback
model.fit(
    train_generator,
    epochs=EPOCHS,
    callbacks=[tensorboard_callback]
)

# Save the trained model
model.save(MODEL_SAVE_PATH)
print(f"Model saved at {MODEL_SAVE_PATH}")
