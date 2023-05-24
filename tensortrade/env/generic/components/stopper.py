'''
At the moment, I'll let the training model to decide when to stop trading and reset the environment.
However, this module will remain here as it is a crucial part of the gymnasium environment.

Ray's Tune has a stopper that can be subclassed to stop the training process.
See default stoppers module for implementation
'''

#from ray.tune import Stopper
import time
from abc import abstractmethod
from typing import Dict, Any
from tensortrade.core.component import Component
from tensortrade.core.base import TimeIndexed


class Stopper(Component, TimeIndexed):
    """A component for determining if the environment satisfies a defined
    stopping criteria.
    """

    registered_name = "stopper"

    @abstractmethod
    def stop(self, env: 'TradingEnv') -> bool:
        """Computes if the environment satisfies the defined stopping criteria.
        Parameters
        ----------
        env : `TradingEnv`
            The trading environment.
        Returns
        -------
        bool
            If the environment should stop or continue.
        """
        raise NotImplementedError()

    def reset(self) -> None:
        """Resets the stopper."""
        pass

class Terminate(Component, TimeIndexed):
    """
    A component for determining if the environment satisfies a defined
    stopping criteria.
    """

    registered_name = "terminator"

    @abstractmethod
    def terminate(self, env: 'TradingEnv') -> bool:
        """
        Computes if the environment satisfies the defined stopping criteria.

        Criteria: 
        Whether the agent reaches the terminal state (as defined under the MDP of the task) which can be positive or negative. An example is reaching the goal state or moving into the lava from the Sutton and Barton, Gridworld. If true, the user needs to call reset().
        Parameters
        ----------
        env : `TradingEnv`
            The trading environment.

        Returns
        -------
        bool
            If the environment should stop or continue.
        """
        raise NotImplementedError()
    def reset(self) -> None:
        """Resets the stopper."""
        pass

class Truncate(Component, TimeIndexed):
    """
    A component for determining if the environment satisfies a defined
    stopping criteria.
    """

    registered_name = "truncater"

    @abstractmethod
    def truncate(self, env: 'TradingEnv') -> bool:
        """Computes if the environment satisfies the defined stopping criteria.

        Criteria:
        Whether the truncation condition outside the scope of the MDP is satisfied. Typically, this is a timelimit, but could also be used to indicate an agent physically going out of bounds. Can be used to end the episode prematurely before a terminal state is reached. If true, the user needs to call reset().
        Parameters
        ----------
        env : `TradingEnv`
            The trading environment.

        Returns
        -------
        bool
            If the environment should stop or continue.
        """

        raise NotImplementedError()
        
    def reset(self) -> None:
        """Resets the stopper."""
        pass
'''
class Combined_Stopper(Terminate, Truncate, Stopper): 
    """
    A component for that inherits from all available stoppers.

    This class enables one to combine several stoppers from different stopper classes.
    
    Eventually, the goal is to make this class the default stopper.
    It's a work in progress

    Notes:
        The classes from which the stoppers inherits from have methods that have to be implemented. This is expected to cause some issues in combining the different stoppers

    Example:

        >>> import numpy as np
        >>> from ray import air, tune
        >>> from ray.air import session
        >>> from ray.tune.stopper import (
        ...     CombinedStopper,
        ...     MaximumIterationStopper,
        ...     TrialPlateauStopper,
        ... )
        >>> from tensortade.env.default.stoppers import (
        ...     TimeLimitStopper,
        ...     RewardThresholdStopper,
        ...     Termintor,
        ...     Truncator)
        >>> stopper = CombinedStopper(
        ...     MaximumIterationStopper(max_iter=20),
        ...     TrialPlateauStopper(metric="my_metric"),
        ...     TimeLimitStopper(time_limit=10),
        ...     RewardThresholdStopper(threshold=0.5),
        ...     Truncator(Truncate, threshold=0.5),
        ... )

    """

    registered_name = "comniner_stopper"

    def __init__(self, 
        *stoppers: Stopper,
        terminate: Terminate,
        truncate: Truncate):

        self._stoppers = stoppers
        self._terminate = terminate
        self._truncate = truncate
    
    def terminate(self, env: 'TradingEnv') -> bool:
        terminate_ = any(te() for te in self._terminate)
        return terminate_

    def truncate(self, env: 'TradingEnv') -> bool:
        truncate_ = any(tr() for tr in self._truncate)
        return truncate_

    def __call__(self, trial_id: str, result: Dict[str, Any], env: 'TradingEnv') -> bool:
        """Returns true if the trial should be terminated given the result."""
        _call = any(s(trial_id, result) for s in self._stoppers)
        return _call

    def stop_all(self, env: 'TradingEnv') -> bool:
        """Returns true if the experiment should be terminated."""
        _stop = any(s.stop_all() for s in self._stoppers)
        return _stop

'''     