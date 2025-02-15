import h5py
import numpy as np
from collections import deque

# Replay Buffer Settings
REPLAY_BUFFER_PATH = "replay_buffer.h5"
REPLAY_BUFFER_SIZE = 50000

class ReplayBuffer:
    def __init__(self, buffer_size=REPLAY_BUFFER_SIZE):
        self.buffer = deque(maxlen=buffer_size)
    
    def add(self, state, action, reward, next_state, done):
        """Adds an experience tuple to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Samples a random batch from the buffer."""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones, dtype=np.float32)
    
    def size(self):
        """Returns the current size of the buffer."""
        return len(self.buffer)
    
    def save(self):
        """Saves the replay buffer to an HDF5 file."""
        with h5py.File(REPLAY_BUFFER_PATH, "w") as f:
            f.create_dataset("states", data=np.array([s for s, _, _, _, _ in self.buffer], dtype='S80'))
            f.create_dataset("actions", data=np.array([a for _, a, _, _, _ in self.buffer], dtype='S6'))
            f.create_dataset("rewards", data=np.array([r for _, _, r, _, _ in self.buffer], dtype=np.float32))
            f.create_dataset("next_states", data=np.array([ns for _, _, _, ns, _ in self.buffer], dtype='S80'))
            f.create_dataset("dones", data=np.array([d for _, _, _, _, d in self.buffer], dtype=np.bool_))
    
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
