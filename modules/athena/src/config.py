import os

# Paths
DATA_DIR = "data/"
TRAIN_DATA = os.path.join(DATA_DIR, "train.h5")
VAL_DATA = os.path.join(DATA_DIR, "val.h5")
TEST_DATA = os.path.join(DATA_DIR, "test.h5")

# Hyperparameters
BATCH_SIZE = 256
EPOCHS = 100
LEARNING_RATE = 3e-4