import tensorflow as tf
import h5py
import numpy as np
import json

with open("move_vocab.json", "r") as f:
    move_vocab = json.load(f)

def move_to_index(move_history, move_vocab, max_sequence_length=50):
    indexed_moves = [move_vocab.get(move.decode("ascii"), 0) for move in move_history] 

    if len(indexed_moves) < max_sequence_length:
        indexed_moves = [0] * (max_sequence_length - len(indexed_moves)) + indexed_moves
    else:
        indexed_moves = indexed_moves[-max_sequence_length:]

    return np.array(indexed_moves, dtype=np.int32)

class HDF5DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, batch_size=32, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.h5_file = h5py.File("training_data/training_data.hdf5", "r")
        self.num_samples = self.h5_file["fens"].shape[0]
        self.indexes = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        """ Returns the number of batches per epoch """
        return int(np.floor(self.num_samples / self.batch_size))  # ✅ Fix: Add this method!

    def __getitem__(self, idx):
        batch_indexes = np.sort(self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size])

        fens = np.array(self.h5_file["fens"][batch_indexes], dtype=np.float32)
        move_histories = np.array(self.h5_file["move_histories"][batch_indexes], dtype="S6")
        turn_indicators = np.array(self.h5_file["turn_indicator"][batch_indexes], dtype=np.float32).reshape(-1, 1)
        eval_scores = np.array(self.h5_file["eval_score"][batch_indexes], dtype=np.float32).reshape(-1, 1)
        legal_moves_mask = np.array(self.h5_file["legal_moves_mask"][batch_indexes], dtype=np.float32)

        move_histories = np.array([move_to_index(history, move_vocab) for history in move_histories], dtype=np.int32)

        return {
            "fen_input": fens,
            "move_seq": move_histories,
            "turn_indicator": turn_indicators,
            "legal_mask": legal_moves_mask,
            "eval_score": eval_scores
        }, None

    def on_epoch_end(self):
        """ Shuffles the dataset at the end of each epoch if enabled """
        if self.shuffle:
            np.random.shuffle(self.indexes)
