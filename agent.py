# St John Grimbly
# 1 May 2021
# Partially inspired by https://github.com/pskrunner14/trading-bot

import random
from typing import List, Union

from networks import QNetwork
from memory import ReplayMemory

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import clone_model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Loss, MeanSquaredError
class DQNAgent:
    """This is core logic for the agent."""

    def __init__(
        self, 
        state_dim: int, 
        action_dim: int = 3, # Default: agents can hold = 0, buy = 1, or sell = 2.
        hidden_layer_sizes: List = [128,256,256,128],
        activation: str = "relu",
        discount: float = 0.99,
        model_name: Union[None, str] = None,
        target_update_freq: int = 10e3,
        learning_rate: float = 1e-3,
        loss: Loss = MeanSquaredError,
        epsilon: float = 0.01, # Probability of random action
        epsilon_decay_rate: float = 1e-5
    ) -> None:
        """Initialise instance of a DQN agent.
            args:
            state_dim: Input size to the agent networks. 
                hidden_layer_size: Sizes and, implicitly, the number of hidden layers.
        """
        self._state_dim = state_dim 
        self._action_dim = action_dim
        self._hidden_layer_sizes = hidden_layer_sizes
        self._activation = activation
        self._discount = discount
        self._model_name = model_name
        self._target_update_freq = target_update_freq
        self._learning_rate = learning_rate
        self._loss = loss
        self._epsilon = epsilon
        self._epsilon_decay_rate = epsilon_decay_rate
        self._action_dim = 3 

        self._iteration = 0
        self._optimizer = Adam(self._learning_rate)        
        self.memory = ReplayMemory(
            self._action_dim
        )
        if self._model_name is not None:
            self.model = self._load()
        else:
            self.model = QNetwork(
                self._state_dim,
                self._action_dim,
                self._hidden_layer_sizes,
                self._activation,
            )
        self.target_model = clone_model(self.model)
    

    def step(self, state, is_training=True) -> None:
        self._iteration += 1
        self._epsilon = self._epsilon * self._epsilon_decay_rate

        if is_training and random.random() < self._epsilon:
            return random.randrange(self._action_dim)
        

        if self._iteration == 1:
            return 1 # Buy on first step

        # Choose action that leads to highest state-action value
        return tf.argmax(self.model(state)) 


    def train(self) -> None:
        if self._iteration % self._target_update_freq == 0:
            self.target_model.set_weights(self.model.get_weights())
        
        batch = self.memory.sample()

        for state, action, reward, next_state, done in batch:
            if done:
                target = reward
            else:
                target = reward + self._discount*tf.reduce_max(self.target_model(next_state))
            q_value = self.model(state)


    def _load_model(self) -> Model:
        """Load a pretrained model if specified."""
        return self.load_model("models/" + self.model_name, custom_objects=self.custom_objects)
       

    #def save(self, episode):
    #    self.model.save("models/{}_{}".format(self.model_name, episode))

    # This code is pytorch code. 
    # def soft_update(self, local_model, target_model, tau):
    #     # θ_target = τ*θ_local + (1 - τ)*θ_target
    #     for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
    #         target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)