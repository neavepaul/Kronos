import h5py
import numpy as np
from collections import deque


from modules.athena.src.game_memory import GameMemory
from modules.athena.src.utils import get_stockfish_eval

REPLAY_BUFFER_PATH = "replay_buffer.h5"
REPLAY_BUFFER_SIZE = 5000
STOCKFISH_DEPTH = 12

class ReplayBuffer:
    def __init__(self, buffer_size=REPLAY_BUFFER_SIZE):
        self.buffer = deque(maxlen=buffer_size)
        self.eval_history = deque(maxlen=buffer_size)
        self.game_memory = GameMemory()
    
    def __len__(self):
        return len(self.buffer)
    
    def add(self, state, action, reward, next_state, done, eval_change):
        """Adds an experience tuple with pattern recall."""
        adjusted_reward = reward + eval_change
        
        self.buffer.append((state, action, adjusted_reward, next_state, done))
        self.eval_history.append(eval_change)

        if done:
            game_moves = [s[1] for s in self.buffer]
            self.game_memory.add_game(game_moves, final_eval=eval_change)


    def sample(self, batch_size):
        """Samples recent, high-reward, and pattern-matching moves."""
        recent_size = int(batch_size * 0.4)
        high_reward_size = int(batch_size * 0.3)
        strategic_size = batch_size - (recent_size + high_reward_size)

        recent_indices = np.random.choice(range(max(0, len(self.buffer) - 5000), len(self.buffer)), recent_size, replace=False)

        rewards = np.array([abs(r) for _, _, r, _, _ in self.buffer])
        prob = rewards / (rewards.sum() + 1e-8)
        high_reward_indices = np.random.choice(len(self.buffer), high_reward_size, replace=False, p=prob)

        eval_improvements = np.diff(self.eval_history, prepend=self.eval_history[0])
        strategic_indices = np.argsort(eval_improvements)[-strategic_size:]

        indices = np.concatenate([recent_indices, high_reward_indices, strategic_indices])
        batch = [self.buffer[i] for i in indices]
        return list(zip(*batch))
    
    def load(self):
        """Loads the replay buffer from an HDF5 file."""
        try:
            with h5py.File(REPLAY_BUFFER_PATH, "r") as f:
                states = f["states"][:]
                actions = f["actions"][:]
                rewards = f["rewards"][:]
                next_states = f["next_states"][:]
                dones = f["dones"][:]
                evals = f["eval_history"][:]  # Load evaluations
                
                self.buffer.extend(zip(states, actions, rewards, next_states, dones))
                self.eval_history.extend(evals)  # Restore evaluation history

            print("Replay buffer loaded from file.")

        except FileNotFoundError:
            print("No existing replay buffer found. Starting fresh.")

        self.game_memory.load()  # Load long-term game memory

def init_replay_buffer():
    """Initializes and loads the replay buffer."""
    buffer = ReplayBuffer()
    buffer.load()
    return buffer
