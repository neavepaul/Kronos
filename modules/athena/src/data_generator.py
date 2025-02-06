import h5py
import numpy as np
import tensorflow as tf

HDF5_FILE = "training_data/training_data.hdf5"
MAX_MOVE_HISTORY = 50  # Fixed move history length

class HDF5DataGenerator(tf.keras.utils.Sequence):
    """
    Data generator that loads training data from an HDF5 file in batches.
    """

    def __init__(self, batch_size=32, shuffle=True, move_vocab=None):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.h5_file = h5py.File(HDF5_FILE, "r")
        self.num_samples = self.h5_file["fens"].shape[0]
        self.indexes = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(self.indexes)

        # Use provided move_vocab StringLookup layer (from model)
        self.move_vocab = move_vocab

    def __len__(self):
        return self.num_samples // self.batch_size

    def __getitem__(self, idx):
        batch_indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_indexes = np.sort(batch_indexes)  # **Fix: Reduce HDF5 access lag**

        # Load batch data from HDF5
        fens = np.array(self.h5_file["fens"][batch_indexes], dtype=np.float32)
        move_histories = np.array(self.h5_file["move_histories"][batch_indexes], dtype="S6")  # Strings (PFFTTT)
        legal_moves_mask = np.array(self.h5_file["legal_moves_mask"][batch_indexes], dtype=np.float32)
        eval_scores = np.array(self.h5_file["eval_score"][batch_indexes], dtype=np.float32).reshape(-1, 1)  # (batch, 1)
        next_moves = np.array(self.h5_file["next_move"][batch_indexes], dtype="S6")  # Next move (PFFTTT)

        # Convert move histories and next moves from byte strings -> normal strings
        move_histories = np.char.decode(move_histories, "ascii")
        next_moves = np.char.decode(next_moves, "ascii")

        # Convert move sequences into indexed integer format
        move_histories = self.move_vocab(move_histories)
        next_moves = self.move_vocab(next_moves)  # Convert labels into integer indices

        return (
            {
                "fen_input": tf.convert_to_tensor(fens),
                "move_seq": tf.convert_to_tensor(move_histories),
                "legal_mask": tf.convert_to_tensor(legal_moves_mask),
                "eval_score": tf.convert_to_tensor(eval_scores),
            },
            {
                "move_probs": tf.one_hot(next_moves, depth=700000),  # One-hot encode next move
                "criticality": tf.zeros((self.batch_size, 1), dtype=tf.float32),  # Placeholder for now
            }
        )

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
