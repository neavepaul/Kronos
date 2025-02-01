import h5py
import numpy as np
import tensorflow as tf

HDF5_FILE = "training_data/training_data.hdf5"

class HDF5DataGenerator(tf.keras.utils.Sequence):
    """
    Data generator that loads training data from an HDF5 file in batches.
    """

    def __init__(self, batch_size=32, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.h5_file = h5py.File(HDF5_FILE, "r")

        self.num_samples = self.h5_file["fens"].shape[0]
        self.indexes = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        """
        Number of batches per epoch.
        """
        return self.num_samples // self.batch_size

    def __getitem__(self, idx):
        """
        Returns a batch of training data.
        """
        batch_indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]

        # Load batch data from HDF5
        fens = self.h5_file["fens"][batch_indexes]
        move_histories = self.h5_file["move_histories"][batch_indexes]
        evaluations = self.h5_file["evaluations"][batch_indexes]
        move_indices_from = self.h5_file["move_indices"][batch_indexes] // 64  # Extract from_square
        move_indices_to = self.h5_file["move_indices"][batch_indexes] % 64  # Extract to_square
        legal_moves_mask = self.h5_file["legal_moves_mask"][batch_indexes]

        return [fens, move_histories, legal_moves_mask], [evaluations, move_indices_from, move_indices_to]

    def on_epoch_end(self):
        """
        Shuffle dataset after each epoch.
        """
        if self.shuffle:
            np.random.shuffle(self.indexes)
