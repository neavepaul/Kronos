import h5py
import numpy as np
from collections import deque

# Replay Buffer Settings
REPLAY_BUFFER_PATH = "replay_buffer.h5"
REPLAY_BUFFER_SIZE = 50000

class ReplayBuffer:
    def __init__(self, buffer_size=REPLAY_BUFFER_SIZE):
        self.buffer = deque(maxlen=buffer_size)
    
    def __len__(self):
        """Allows `len(replay_buffer)` to return the current buffer size."""
        return len(self.buffer)
    
    def add(self, state, action, reward, next_state, done):
        """Adds an experience tuple to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Samples a random batch from the buffer."""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return list(states), list(actions), list(rewards), list(next_states), list(dones)
    
    def size(self):
        """Returns the current size of the buffer."""
        return len(self.buffer)
    
    def save(self):
        """Saves the replay buffer to an HDF5 file."""
        with h5py.File(REPLAY_BUFFER_PATH, "w") as f:
            # Ensure FEN states are stored as strings (not tensors)
            f.create_dataset("states", data=np.array([str(s[0]) for s, _, _, _, _ in self.buffer], dtype="S80"))
            
            # Store move history as int arrays
            move_histories = np.stack([s[1] for s, _, _, _, _ in self.buffer])
            f.create_dataset("move_histories", data=move_histories)
            
            # Store legal move masks
            legal_moves_masks = np.stack([s[2] for s, _, _, _, _ in self.buffer])
            f.create_dataset("legal_moves_masks", data=legal_moves_masks)

            # Store turn indicators
            turn_indicators = np.stack([s[3] for s, _, _, _, _ in self.buffer])
            f.create_dataset("turn_indicators", data=turn_indicators)

            # Actions (stored as strings)
            f.create_dataset("actions", data=np.array([a for _, a, _, _, _ in self.buffer], dtype="S6"))

            # Rewards (ensure float format)
            f.create_dataset("rewards", data=np.array([float(r) for _, _, r, _, _ in self.buffer], dtype=np.float32))

            # Next states (ensure FEN strings)
            f.create_dataset("next_states", data=np.array([str(ns[0]) for _, _, _, ns, _ in self.buffer], dtype="S80"))

            # Done flags (convert to boolean)
            f.create_dataset("dones", data=np.array([bool(d) for _, _, _, _, d in self.buffer], dtype=np.bool_))

    def load(self):
        """Loads the replay buffer from an HDF5 file."""
        try:
            with h5py.File(REPLAY_BUFFER_PATH, "r") as f:
                states = f["states"][:]
                actions = f["actions"][:]
                rewards = f["rewards"][:]
                next_states = f["next_states"][:]
                dones = f["dones"][:]
                self.buffer.extend(zip(states, actions, rewards, next_states, dones))
            print("Replay buffer loaded from file.")
        except FileNotFoundError:
            print("No existing replay buffer found. Starting fresh.")

def init_replay_buffer():
    """Initializes and loads the replay buffer."""
    buffer = ReplayBuffer()
    buffer.load()
    return buffer
