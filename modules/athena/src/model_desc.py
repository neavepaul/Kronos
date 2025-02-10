import tensorflow as tf
from keras.layers import TFSMLayer

# Define model directory
model_path = "models"  # Adjust path if necessary

# Load the model using TFSMLayer (for inference)
model = TFSMLayer(model_path, call_endpoint="serving_default")

# Check if it's loaded correctly
print(model)
model.summary()
