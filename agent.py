# St John Grimbly
# 1 May 2021
# Partially inspired by https://github.com/pskrunner14/trading-bot

import random
from typing import List, Union
from collections import deque

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import clone_model, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Loss, MeanSquaredError
class DQNAgent:
    """This is core logic for the agent."""

    def __init__(
        self, 
        state_dim: int, 
        hidden_layer_sizes: List = [128,256,256,128],
        activation: str = "relu",
        discount: float = 0.99,
        model_name: Union[None, str] = None,
        target_update_freq: int = 10e3,
        learning_rate: float = 1e-3,
        loss: Loss = MeanSquaredError,
    ) -> None:
        """Initialise instance of a DQN agent.
            args:
            state_dim: Input size to the agent networks. 
                hidden_layer_size: Sizes and, implicitly, the number of hidden layers.
        """
        self._state_dim = state_dim 
        self._hidden_layer_sizes = hidden_layer_sizes
        self._activation = activation
        self._discount = discount
        self._model_name = model_name
        self._target_update_freq = target_update_freq
        self._learning_rate = learning_rate
        self._loss = loss
        
        self._optimizer = Adam(self._learning_rate)
        self._target_trailing_by = 0
        self.memory = deque(maxlen=1000)
        self.num_actions = 3 # Agents can hold, buy, or sell.
        if self._model_name is not None:
            self.model = self._load()
        else:
            self.model = self._q_network()
        self.target_model = clone_model(self.model)
    
    def _load_model(self) -> Model:
        """Load a pretrained model if specified."""
        return self.load_model("models/" + self.model_name, custom_objects=self.custom_objects)

    def _q_network(self) -> Model:
        """Create the neural network architecture for the DQN agent."""
        model = tf.keras.Sequential()
        model.add(Dense(units=self._hidden_layer_sizes[0], 
                        input_dim=self._state_dim, 
                        activation=self._activation))

        for i in range(2,len(self._hidden_layer_sizes)):
            model.add(Dense(self._hidden_layer_sizes[i], activation=self._activation))

        model.add(Dense(self.num_actions))
        model.compile(loss=self._loss, optimizer=self._optimizer)
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

    def train(self, batch_size: int = 32) -> None:
        if self._target_trailing_by % self._target_update_freq == 0:
            self.target_model.set_weights(self.model.get_weights())
        
        batch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in batch:
            if done:
                target = reward
            else:
                target = reward + self._discount*tf.argmax(self.target_model(next_state)[0])

    #def save(self, episode):
    #    self.model.save("models/{}_{}".format(self.model_name, episode))