import os
from datetime import datetime
import tensorflow as tf
from model import get_model
from data_generator import HDF5DataGenerator

BATCH_SIZE = 64
EPOCHS = 1

# Define the model save path including the timestamp and number of epochs
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
MODEL_SAVE_PATH = f"models/athena_gm_trained_{timestamp}_{EPOCHS}epochs.h5"

os.makedirs("models", exist_ok=True)
train_generator = HDF5DataGenerator(batch_size=BATCH_SIZE)
model = get_model()

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=MODEL_SAVE_PATH, save_best_only=True, monitor="loss", mode="min")

model.fit(train_generator, epochs=EPOCHS, callbacks=[checkpoint_callback])
model.save(MODEL_SAVE_PATH)
