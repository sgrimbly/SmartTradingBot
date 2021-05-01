import random
from collections import deque, namedtuple
from typing import NamedTuple

import tensorflow as tf


class ReplayMemory:
    """Fixed-size buffer to store experience tuples."""

    def __init__(
        self,
        action_dim,
        buffer_size: int = 1000,
        batch_size: int = 32,
        seed: int = 0,  # For debugging/testing
    ) -> None:
        self._action_dim = action_dim
        self._memory = deque(maxlen=buffer_size)
        self._batch_size = batch_size
        self._experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"],
        )
        self._seed = seed

    def add(
        self,
        state: tf.Tensor,
        action: tf.Tensor,
        reward: tf.Tensor,
        next_state: tf.Tensor,
        done: bool,
    ) -> None:
        """Add memory to replay buffer."""
        experience = self.experience(state, action, reward, next_state, done)
        self._memory.append(experience)

    def sample(self) -> NamedTuple:
        experience = random.sample(self._memory, k=self.batch_size)
        return experience

    def __len__(self):
        return len(self.memory)
