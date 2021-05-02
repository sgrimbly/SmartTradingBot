# St John Grimbly
# 1 May 2021
# Partially inspired by https://github.com/pskrunner14/trading-bot

import random
from typing import List, Optional, Union

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import clone_model, load_model
from tensorflow.keras.optimizers import Adam

from SmartTradingBot import memory, networks


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
        model_name: Optional[str] = None,
        learning_freq: int = 4,
        target_update_freq: int = 1000,
        learning_rate: float = 1e-3,
        loss: str = "mse",
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

        self._iteration = 1
        self._optimizer = Adam(self._learning_rate)
        self.memory = memory.ReplayMemory(
            self._action_dim,
            self._buffer_size,
            self._batch_size,
        )
        # self.model = ""
        if self._model_name is not None:
            self.model = self._load_model()
        else:
            q_network = networks.QNetwork(
                self._state_dim,
                self._action_dim,
                self._hidden_layer_sizes,
                self._activation,
            )
            self.model = q_network.get_model()
        self.model.compile(self._optimizer, loss=self._loss)
        self.target_model = clone_model(self.model)

    def act(self, state: tf.Tensor, evaluation: bool = False) -> float:
        """Choose an action without learning."""
        # if self._iteration == 1:
        #    return 1  # Buy on first step
        if random.random() < self._epsilon and not evaluation:
            return random.randrange(self._action_dim)
        # Choose action that leads to highest state-action value
        return tf.argmax(self.model(state)[0]).numpy()

    def step(
        self,
        state: tf.Tensor,
        action: tf.Tensor,
        reward: float,
        next_state: tf.Tensor,
        done: bool,
    ) -> Union[float, bool]:
        """Experience a step in the environment."""
        self._iteration += 1
        self.memory.add(state, action, reward, next_state, done)

        # Decay random action probability
        self._epsilon = self._epsilon * self._epsilon_decay_rate

        if (
            self._iteration % self._learning_freq == 0
            and len(self.memory) > self._batch_size
        ):
            return self.learn()
        return False

    def learn(self) -> float:
        minibatch = self.memory.sample()

        if self._iteration % self._target_update_freq == 0:
            self.target_model.set_weights(self.model.get_weights())

        X_train, y_train = [], []

        for state, action, reward, next_state, done in minibatch:
            if done:
                target = reward
            else:
                target = reward + self._discount * tf.reduce_max(
                    self.target_model(next_state)[0]
                )
            q_values = self.model.predict(state)
            q_values[0][action] = target

            X_train.append(state[0])
            y_train.append(q_values[0])

        fit = self.model.fit(np.array(X_train), np.array(y_train))
        return fit.history["loss"][0]

    def _load_model(self) -> Model:
        """Load a pretrained model if specified."""
        return load_model(
            "models/" + self._model_name, custom_objects={}  # type: ignore
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
