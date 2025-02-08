# q_network.py
import tensorflow as tf
from tensorflow.keras import Model, layers

class QNetwork(Model):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.dense1 = layers.Dense(512, activation="relu")
        self.dense2 = layers.Dense(256, activation="relu")
        self.dense3 = layers.Dense(128, activation="relu")
        self.q_values = layers.Dense(6, activation="linear")  # Output PFFTTU (Piece, From, To, To, Upgrade)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return self.q_values(x)
