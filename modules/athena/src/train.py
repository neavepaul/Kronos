import tensorflow as tf
from model import get_model  # Import the function, not the model object
from data_generator import HDF5DataGenerator

# Training parameters
BATCH_SIZE = 32
EPOCHS = 10
MODEL_SAVE_PATH = "models/athena_trained.h5"

# Initialize model
model = get_model()

# Initialize Data Generator
train_generator = HDF5DataGenerator(batch_size=BATCH_SIZE)

# Train the model
model.fit(
    train_generator,
    epochs=EPOCHS
)

# Save the trained model
model.save(MODEL_SAVE_PATH)
print(f"Model saved at {MODEL_SAVE_PATH}")
