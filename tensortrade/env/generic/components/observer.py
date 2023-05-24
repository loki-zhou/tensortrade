
from abc import abstractmethod

import numpy as np
from gymnasium.spaces import Space

from tensortrade.core.base import TimeIndexed
from tensortrade.core.component import Component


class Observer(Component, TimeIndexed):
    """A component to generate an observation at each step of an episode.
    """

    registered_name = "observer"

    @property
    @abstractmethod
    def observation_space(self) -> Space:
        """The observation space of the `TradingEnv`. (`Space`, read-only)
        """
        raise NotImplementedError()

    @abstractmethod
    def observe(self, env: 'TradingEnv') -> np.array:
        """Gets the observation at the current step of an episode

        Parameters
        ----------
        env: 'TradingEnv'
            The trading environment.

        Returns
        -------
        `np.array`
            The current observation of the environment.
        """
        raise NotImplementedError()

    def reset(self, random_start=0):
        """Resets the observer."""
        pass
