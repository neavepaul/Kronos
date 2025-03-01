import h5py
import numpy as np
from collections import deque

from modules.athena.src.game_memory import GameMemory

REPLAY_BUFFER_PATH = "replay_buffer.h5"
REPLAY_BUFFER_SIZE = 5000

class ReplayBuffer:
    def __init__(self, buffer_size=REPLAY_BUFFER_SIZE):
        self.buffer = deque(maxlen=buffer_size)
        self.game_memory = GameMemory()
    
    def __len__(self):
        return len(self.buffer)
    
    def add(self, state, action, reward, next_state, done, fen):
        """Adds an experience tuple with pattern recall."""
        self.buffer.append((state, action, reward, next_state, done))

        if done:
            game_moves = [s[1] for s in self.buffer]
            self.game_memory.add_game(game_moves, final_fen=fen)
    
    def sample(self, batch_size):
        """Samples recent, high-reward, and pattern-matching moves."""
        recent_size = int(batch_size * 0.4)
        high_reward_size = int(batch_size * 0.3)
        strategic_size = batch_size - (recent_size + high_reward_size)

        recent_indices = np.random.choice(range(max(0, len(self.buffer) - 5000), len(self.buffer)), recent_size, replace=False)

        rewards = np.array([r for _, _, r, _, _ in self.buffer])
        prob = rewards / (rewards.sum() + 1e-8)
        high_reward_indices = np.random.choice(len(self.buffer), high_reward_size, replace=False, p=prob)

        indices = np.concatenate([recent_indices, high_reward_indices, np.random.choice(len(self.buffer), strategic_size, replace=False)])
        samples = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*samples)
        return states, actions, rewards, next_states, dones
    
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

        self.game_memory.load()  # Load long-term game memory

def init_replay_buffer():
    """Initializes and loads the replay buffer."""
    buffer = ReplayBuffer()
    buffer.load()
    return buffer
