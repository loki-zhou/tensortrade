
import logging
import uuid

from typing import Any, Dict, Optional, Tuple, List
from pandas import DataFrame
import gymnasium as gym
import numpy as np
from gymnasium.utils import seeding#, save_video

from tensortrade.core import Clock, Component, TimeIndexed
from tensortrade.env.generic import (ActionScheme, Informer, Observer,
                                    Renderer, RewardScheme, Stopper)

'''
The main Gymnasium class for implementing Reinforcement Learning Agents environments.

The class encapsulates an environment with arbitrary behind-the-scenes dynamics through the step and reset functions. An environment can be partially or fully observed by single agents. For multi-agent environments, see PettingZoo.

The main API methods that users of this class need to know are:

step - Updates an environment with actions returning the next agent observation, the reward for taking that actions, if the environment has terminated or truncated due to the latest action and information from the environment about the step, i.e. metrics, debug info.
reset - Resets the environment to an initial state, required before calling step. Returns the first agent observation for an episode and information, i.e. metrics, debug info.
render - Renders the environments to help visualise what the agent see, examples modes are "human", "rgb_array", "ansi" for text.
close - Closes the environment, important when external software is used, i.e. pygame for rendering, databases
Environments have additional attributes for users to understand the implementation

action_space - The Space object corresponding to valid actions, all valid actions should be contained within the space.
observation_space - The Space object corresponding to valid observations, all valid observations should be contained within the space.
reward_range - A tuple corresponding to the minimum and maximum possible rewards for an agent over an episode. The default reward range is set to (-infty,+infty).
spec - An environment spec that contains the information used to initialize the environment from gymnasium.make
metadata - The metadata of the environment, i.e. render modes, render fps
np_random - The random number generator for the environment. This is automatically assigned during super().reset(seed=seed) and when assessing self.np_random.

'''
class TradingEnv(gym.Env, TimeIndexed):
    """A trading environment made for use with Gym-compatible reinforcement
    learning algorithms.

    Parameters
    ----------
    action_scheme : `ActionScheme`
        A component for generating an action to perform at each step of the
        environment.
    reward_scheme : `RewardScheme`
        A component for computing reward after each step of the environment.
    observer : `Observer`
        A component for generating observations after each step of the
        environment.
    informer : `Informer`
        A component for providing information after each step of the
        environment.
    renderer : `Renderer`
        A component for rendering the environment.
    kwargs : keyword arguments
        Additional keyword arguments needed to create the environment.
    """

    agent_id: str = None
    episode_id: str = None

    def __init__(
        self,
        action_scheme: ActionScheme,
        reward_scheme: RewardScheme,
        observer: Observer,
        stopper: Stopper,
        informer: Informer,
        renderer: Renderer,
        min_periods: int = None,
        max_episode_steps: int = None,

        random_start_pct: float = 0.00,
        starting_point: bool = True,
        **kwargs) -> None:
        super().__init__()
        self.clock = Clock()

        self.action_scheme = action_scheme
        self.reward_scheme = reward_scheme
        self.observer = observer
        self.stopper = stopper
        self.informer = informer
        self.renderer = renderer
        self.min_periods = min_periods
        self.random_start_pct = random_start_pct
        self.starting_point = starting_point # I think this should go into the observer

        # Register the environment in Gym and fetch spec
        """
        Registers an environment in gymnasium with an ``id`` to use with :meth:`gymnasium.make` with the ``entry_point`` being a string or callable for creating the environment.
        The ``id`` parameter corresponds to the name of the environment, with the syntax as follows:
        ``[namespace/](env_name)[-v(version)]`` where ``namespace`` and ``-v(version)`` is optional.
        It takes arbitrary keyword arguments, which are passed to the :class:`EnvSpec` ``kwargs`` parameter.
        Args:
            id: The environment id
            entry_point: The entry point for creating the environment
            reward_threshold: The reward threshold considered for an agent to have learnt the environment
            nondeterministic: If the environment is nondeterministic (even with knowledge of the initial seed and all actions, the same state cannot be reached)
            max_episode_steps: The maximum number of episodes steps before truncation. Used by the :class:`gymnasium.wrappers.TimeLimit` wrapper if not ``None``.
            order_enforce: If to enable the order enforcer wrapper to ensure users run functions in the correct order.
                If ``True``, then the :class:`gymnasium.wrappers.OrderEnforcing` is applied to the environment.
            autoreset: If to add the :class:`gymnasium.wrappers.AutoResetWrapper` such that on ``(terminated or truncated) is True``, :meth:`gymnasium.Env.reset` is called.
            disable_env_checker: If to disable the :class:`gymnasium.wrappers.PassiveEnvChecker` to the environment.
            apply_api_compatibility: If to apply the :class:`gymnasium.wrappers.StepAPICompatibility` wrapper to the environment.
                Use if the environment is implemented in the gym v0.21 environment API.
            vector_entry_point: The entry point for creating the vector environment
            **kwargs: arbitrary keyword arguments which are passed to the environment constructor on initialisation.
        """
        id = 'LanuvoTrade-v1.0'
        gym.envs.register(
            id=id,
            max_episode_steps=max_episode_steps,
            apply_api_compatibility=True, # For compat issues Ray
            disable_env_checker=True, # Active on rllib
            entry_point= None,
            #reward_threshold = None,
            #nondeterministic = False,
            #order_enforce= True,
            autoreset = False,
        )
        self.spec = gym.spec(env_id=id)

        for c in self.components.values():
            c.clock = self.clock

        self.action_space = action_scheme.action_space
        self.observation_space = observer.observation_space

        self._enable_logger = kwargs.get('enable_logger', False)
        if self._enable_logger:
            self.logger = logging.getLogger(kwargs.get('logger_name', __name__))
            self.logger.setLevel(kwargs.get('log_level', logging.DEBUG))

    @property
    def components(self) -> 'Dict[str, Component]':
        """The components of the environment. (`Dict[str,Component]`, read-only)"""
        return {
            "action_scheme": self.action_scheme,
            "reward_scheme": self.reward_scheme,
            "observer": self.observer,
            "stopper": self.stopper,
            "informer": self.informer,
            "renderer": self.renderer
        }

    def step(self, action: Any) -> 'Tuple[np.array, float, bool, dict]':
        """Makes one step through the environment.

        Parameters
        ----------
        action : Any
            An action to perform on the environment.

        Returns
        -------
        `np.array`
            The observation of the environment after the action being
            performed.
        `float`
            The computed reward for performing the action.
        `bool`
            Whether or not the episode is complete.
        `dict`
            The information gathered after completing the step.
        """
        self.action_scheme.perform(self, action)
        reward = self.reward_scheme.reward(self) 
        obs = self.observer.observe(self)
        info = self.informer.info(self)

        truncated = False # self.stopper.stop(self)
        terminated = self.stopper.stop(self)

        self.clock.increment()

        return obs, reward, terminated, truncated, info

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Resets the environment to an initial internal state, returning an initial `observation` and `info`.

        This method generates a new starting state often with some randomness to ensure that the agent explores the state space and learns a generalised policy about the environment. This randomness can be controlled with the `seed` parameter, otherwise if the environment already has a random number generator and `reset` is called with `seed=None`, the RNG is not reset.

        Therefore, `reset` should (in the typical use case) be called with a `seed` right after initialization and then never again.

        For Custom environments, the first line of reset should be `super().reset(seed=seed)` which implements the seeding correctly by setting the np_random with the value of `seed`. 
        To update older environments, gymnasium highly recommends that `super().reset(seed=seed)` is called on the first line of reset().
        
        Note: This call to super does NOT return anything. Therefore, seed the PRNG of this space. This removes the need for building a separate automatic update method of np_random with seed value.

        _____________________

        Args:
            `seed` (optional int): The seed that is used to initialize the environment's PRNG (np_random).
                If the environment does not already have a PRNG and `seed=None` (the default option) is passed, a seed will be chosen from some source of entropy (e.g. timestamp or /dev/urandom). 
                However, if the environment already has a PRNG and `seed=None` is passed, the PRNG will *not* be reset. 
                If you pass an integer, the PRNG will be reset even if it already exists. 
                Usually, you want to pass an integer *right after the environment has been initialized and then never again*. Please refer to the minimal example above to see this paradigm in action.
            `options` (optional dict): Additional information to specify how the environment is reset (optional,
                depending on the specific environments
        _____________________

        Returns:
            `observation` (ObsType):  `np.array` or DataFrame
                Observation of the initial state( The first observation of the environment). This will be an element of `observation_space` (typically a numpy array) and is analogous to the observation returned by `step`.
            `info` (dictionary): This dictionary contains auxiliary information complementing observation. It should be analogous to the `info` returned by `step`.
        _____________________

        PRNG:
        
        Returns - 
        `out` : 
        `int` or `ndarray` of ints size-shaped array of random integers from the appropriate distribution, or a single such random int if size not provided.

        The code checks if the `random_start_pct` is greater than 0.00. If it is, it generates a random start index based on the size of the iterable in the observer's feed process.
        _____________________
    
        The code then generates a unique episode ID and resets the clock.
        
        It then goes through each component and calls its reset function if it has one, passing in the random start index for the observer only.
        
        Finally, it observes the environment and increments the clock before returning the initial observation and info
        _________________________________________________

        Ray's Rllib Warning
        WARNING env.py:166 -- Your env reset() method appears to take 'seed' or 'return_info' arguments. Note that these are not yet supported in RLlib. Seeding will take place using 'env.seed()' and the info dict will not be returned from reset.
        """
        super().reset(seed=seed)
        #obs, info = super().reset(seed=seed, options=options) 
        # TypeError: cannot unpack non-iterable NoneType object - Why is this failing?? I could use this to reset obs and info
        
        #self.env = gym.make("CartPole-v1")
        #self.spec = gym.make("CartPole-v1")
        #obs, info = self.env.reset(seed=seed, options=options)

        # RNG
        reset_seed = seed
        np_seed = self.seed(seed=reset_seed)
        random_start = self.random_start(seed=np_seed)

        self.episode_id = str(uuid.uuid4())
        self.clock.reset()

        for c in self.components.values():
            if hasattr(c, "reset"):
                if isinstance(c, Observer):
                    c.reset(random_start=random_start)
                else:
                    c.reset()

        # The below lines should return enpty arrays and dict. 
        # TODO: COnfirm that this happens or find a way to return the reset versions 
        obs = self.observer.observe(self)
        info = self.informer.info(self)

        self.clock.increment()

        return obs, info 

    def render(self, **kwargs) -> None:
        """Renders the environment."""
        self.renderer.render(self, **kwargs)

    def save(self) -> None:
        """Saves the rendered view of the environment."""
        self.renderer.save()

    def close(self) -> None:
        """Closes the environment."""
        self.renderer.close()
    
    def seed(self, seed: Optional[int]= None)-> Any:
        '''
        deprecated: in favor of Env.reset(seed=seed).

        _______

        To update older environments, gymnasium highly recommends that super().reset(seed=seed) is called on the first line of reset(). This will automatically update the np_random with the seed value.

        _______

        Ray's Rllib Warning
        WARNING env.py:166 -- Your env reset() method appears to take 'seed' or 'return_info' arguments. Note that these are not yet supported in RLlib. Seeding will take place using 'env.seed()' and the info dict will not be returned from reset.
        '''
        rng, np_seed = seeding.np_random(seed)
        rng
        return np_seed
        #return [np_seed]
    
    def random_start(self, seed: Optional[int]= None):
        '''
        In house PRNG

        Note that if you use custom reset bounds, it may lead to out-of-bound state/observations.
        '''
        size = len(self.observer.feed.process[-1].inputs[0].iterable)
        # TODO: EXplore effects of auto seeding
        if self.random_start_pct > 0.00:
            random_start = self.np_random.integers(
                low= 0, 
                high= int(size * self.random_start_pct))
        elif self.random_start_pct == 0.00 and seed is not None:
            random_start = self.np_random.integers(
                low= 0, 
                high= int(size * seed))
        elif self.random_start_pct == 0.00 and seed == None:
            random_start = 0
        return random_start
        
    