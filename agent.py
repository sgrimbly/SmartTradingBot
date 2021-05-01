# St John Grimbly
# 1 May 2021
# Partially inspired by https://github.com/pskrunner14/trading-bot

import random
from typing import List, Union

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.losses import Loss, MeanSquaredError
from tensorflow.keras.models import clone_model, load_model
from tensorflow.keras.optimizers import Adam

from memory import ReplayMemory
from networks import QNetwork


class DQNAgent:
    """This is core logic for the agent."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 3,  # Default: agents can hold=0, buy=1, or sell=2.
        hidden_layer_sizes: List = [128, 256, 256, 128],
        activation: str = "relu",
        discount: float = 0.99,
        batch_size: int = 32,
        buffer_size: int = 1000,
        model_name: Union[None, str] = None,
        learning_freq: int = 4,
        target_update_freq: int = 10e3,
        learning_rate: float = 1e-3,
        loss: Loss = MeanSquaredError,
        epsilon: float = 0.01,  # Probability of random action
        epsilon_decay_rate: float = 1e-5,
    ) -> None:
        """Initialise instance of a DQN agent.
        args:
        state_dim: Input size to the agent networks.
            hidden_layer_size: Sizes and, implicitly, number hidden layers.
        """
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._hidden_layer_sizes = hidden_layer_sizes
        self._activation = activation
        self._discount = discount
        self._batch_size = batch_size
        self._buffer_size = buffer_size
        self._model_name = model_name
        self._learning_freq = learning_freq
        self._target_update_freq = target_update_freq
        self._learning_rate = learning_rate
        self._loss = loss
        self._epsilon = epsilon
        self._epsilon_decay_rate = epsilon_decay_rate
        self._action_dim = 3

        self._iteration = 0
        self._optimizer = Adam(self._learning_rate)
        self.memory = ReplayMemory(
            self._action_dim,
            self._buffer_size,
            self._batch_size,
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

    def act(self, state: tf.Tensor, evaluation: bool = False):
        """Choose an action without learning."""
        if self._iteration == 1:
            return 1  # Buy on first step
        elif random.random() < self._epsilon and not evaluation:
            return random.randrange(self._action_dim)
        # Choose action that leads to highest state-action value
        return tf.argmax(self.model(state))

    def step(
        self,
        state: tf.Tensor,
        action: tf.Tensor,
        reward: float,
        next_state: tf.Tensor,
        done: bool,
    ) -> None:
        """Experience a step in the environment."""
        self._iteration += 1
        self.memory.add(state, action, reward, next_state, done)

        if (
            self._iteration % self._learning_freq == 0
            and len(self.memory) > self._batch_size
        ):
            self.learn()

        # Decay random action probability
        self._epsilon = self._epsilon * self._epsilon_decay_rate

    def learn(self) -> float:
        minibatch = self.memory.sample()

        if self._iteration % self._target_update_freq == 0:
            self.target_model.set_weights(self.model.get_weights())

        for state, action, reward, next_state, done in minibatch:
            if done:
                target = reward
            else:
                target = reward + self._discount * tf.reduce_max(
                    self.target_model(next_state)
                )
            q_value = tf.reduce_max(self.model(state))

            loss = self._loss(target, q_value).numpy()
            return loss

    def _load_model(self) -> Model:
        """Load a pretrained model if specified."""
        return load_model(
            "models/" + self.model_name, custom_objects=self.custom_objects
        )

    # def save(self, episode):
    #    self.model.save("models/{}_{}".format(self.model_name, episode))

    # This code is pytorch code.
    # def soft_update(self, local_model, target_model, tau):
    #     # θ_target = τ*θ_local + (1 - τ)*θ_target
    #     for target_param, local_param in zip(
    #                   target_model.parameters(), local_model.parameters()):
    #         target_param.data.copy_(tau*local_param.data +
    # (1.0-tau)*target_param.data)
