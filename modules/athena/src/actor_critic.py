# actor_critic.py
import tensorflow as tf
from tensorflow.keras import Model, layers

class ActorCritic(Model):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.shared = layers.Dense(512, activation="relu")
        self.actor = layers.Dense(6, activation="softmax")  # Predict move probabilities (PFFTTU)
        self.critic = layers.Dense(1, activation="linear")  # Predict state value (V(s))

    def call(self, inputs):
        x = self.shared(inputs)
        return self.actor(x), self.critic(x)  # Returns move probs & state value
