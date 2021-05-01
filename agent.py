# St John Grimbly
# 1 May 2021
# Partially inspired by https://github.com/pskrunner14/trading-bot

from typing import List, Union
from collections import deque

import tensorflow as tf
from tensorflow.keras.layers import Dense

class DQNAgent:
    """This is core logic for the agent."""

    def __init__(
        self, 
        state_dim: int, 
        hidden_layer_sizes = [128,256,256,128]: List,
        activation = "relu": str,
        model_name = Union[None, str]: str,
    ) -> None:
        """Initialise instance of a DQN agent.
            args:
            state_dim: Input size to the agent networks. 
                hidden_layer_size: Sizes and, implicitly, the number of hidden layers.
        """
        self._state_dim = state_dim 
        self._hidden_layer_sizes = hidden_layer_sizes
        self._activation = activation
        self._model_name = model_name

        self.memory = deque(maxlen=10e4)
        self.num_actions = 3 # Agents can hold, buy, or sell.
        if self._model_name is not None:
            self.model = self._load()
        else:
            self.model = self._model()
    
    def _load_model(self) -> tf.Model:
        """Load a pretrained model if specified."""
        return self._load_model("models/" + self.model_name, custom_objects=self.custom_objects)

    def _model(self) -> tf.Model:
        """Create the neural network architecture for the DQN agent."""
        model = tf.keras.Sequential()
        model.add(Dense(units=self._hidden_layer_sizes[0], 
                        input_dim=self._state_dim, 
                        activation=self._activation))

        for i in range(2,len(self._hidden_layer_sizes)):
            model.add(Dense(self._hidden_layer_sizes[i], activation=self._activation))

        model.add(Dense(self.num_actions))
        return model

    def add_memory(
        self, 
        state: tf.Tensor, 
        action: tf.Tensor, 
        reward: tf.Tensor, 
        next_state: tf.Tensor, 
        done: bool
    ) -> None:
        """Add memory to replay buffer."""
        self.memory.append((state, action, reward, next_state, done))

    