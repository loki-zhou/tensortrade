
from abc import abstractmethod, ABCMeta
from typing import Any
from enum import Enum
from gymnasium.spaces import Space


from tensortrade.core.component import Component
from tensortrade.core.base import TimeIndexed


class ActionScheme(Component, TimeIndexed, metaclass=ABCMeta):
    """A component for determining the action to take at each step of an
    episode.
    """

    registered_name = "actions"

    @property
    @abstractmethod
    def action_space(self) -> Space:
        """The action space of the `TradingEnv`. (`Space`, read-only)
        """
        raise NotImplementedError()

    @abstractmethod
    def perform(self, env: 'TradingEnv', action: Any) -> None:
        """Performs an action on the environment.

        Parameters
        ----------
        env : `TradingEnv`
            The trading environment to perform the `action` on.
        action : Any
            The action to perform on `env`.
        """
        raise NotImplementedError()

    def reset(self) -> None:
        """Resets the action scheme."""
        pass


class BaseActions(Enum):
    """
    Default action space, mostly used for type handling.
    """
    Neutral = 0
    Long_enter = 1
    Long_exit = 2
    Short_enter = 3
    Short_exit = 4


class Positions(Enum):
    Short = 0
    Long = 1
    Neutral = 0.5

    def opposite(self):
        return Positions.Short if self == Positions.Long else Positions.Long

