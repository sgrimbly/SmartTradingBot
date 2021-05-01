from typing import List

import tensorflow as tf
from tensorflow.keras.layers import Dense

"""Possible agent network structures implemented as Tensorflow Modules"""


class QNetwork(tf.Module):
    """Create the neural network architecture for the DQN agent."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 3,  # Default: agents can hold=0, buy=1, or sell=2.
        hidden_layer_sizes: List = [128, 256, 256, 128],
        activation: str = "relu",
    ):
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._hidden_layer_sizes = hidden_layer_sizes
        self._activation = activation

        self._model = tf.keras.Sequential()
        self._model.add(
            Dense(
                units=self._hidden_layer_sizes[0],
                input_dim=self._state_dim,
                activation=self._activation,
            )
        )

        for i in range(2, len(self._hidden_layer_sizes)):
            self._model.add(
                Dense(self._hidden_layer_sizes[i], activation=self._activation)
            )

        self._model.add(Dense(self._action_dim))

    def __call__(self, states: tf.Tensor) -> tf.Tensor:
        return self._model(states)
