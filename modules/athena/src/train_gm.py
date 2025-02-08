# train_gm.py
import os
import tensorflow as tf
from model import get_model
from data_generator import HDF5DataGenerator

BATCH_SIZE = 64
EPOCHS = 10  
MODEL_SAVE_PATH = "models/athena_gm_trained.h5"

os.makedirs("models", exist_ok=True)
train_generator = HDF5DataGenerator(batch_size=BATCH_SIZE)
model = get_model()

# Test model inference on a batch BEFORE training starts
dummy_batch = next(iter(train_generator))
dummy_inputs, _ = dummy_batch

print("\n🔍 Debugging Model Input Shapes Before Training:")
for key, value in dummy_inputs.items():
    print(f"{key}: shape {value.shape}")

# Try a forward pass with the model
try:
    print("\n🚀 Running model on dummy batch...")
    dummy_output = model(dummy_inputs)  # Forward pass
    print("✅ Model inference successful!")
except Exception as e:
    print(f"🚨 Model error: {e}")



checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=MODEL_SAVE_PATH, save_best_only=True, monitor="loss", mode="min")

model.fit(train_generator, epochs=EPOCHS, callbacks=[checkpoint_callback])
model.save(MODEL_SAVE_PATH)
