# Copyright 2019 The TensorTrade Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import gymnasium
import json

import pandas as pd
import numpy as np

from abc import ABCMeta, abstractmethod
from typing import Union, Callable, List, Dict

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.base_class import BaseAlgorithm as BaseRLModel
from stable_baselines3 import PPO


from tensortrade.environments.trading_environment import TradingEnvironment
from tensortrade.strategies import TradingStrategy


class StableBaselinesTradingStrategy(TradingStrategy):
    """A trading strategy capable of self tuning, training, and evaluating with stable-baselines.

    Arguments:
        environments: An instance of a trading environments for the agent to trade within.
        model: The RL model to create the agent with.
            Defaults to DQN.
        policy: The RL policy to train the agent's model with.
            Defaults to 'MlpPolicy'.
        model_kwargs: Any additional keyword arguments to adjust the model.
        kwargs: Optional keyword arguments to adjust the strategy.
    """

    def __init__(self,
                 environment: TradingEnvironment,
                 model: BaseRLModel = PPO,
                 policy: Union[str, BasePolicy] = 'MlpPolicy',
                 model_kwargs: any = {},
                 **kwargs):
        self._model = model
        self._model_kwargs = model_kwargs

        self.environment = environment
        self._agent = self._model(policy, self._environment, **self._model_kwargs)

    @property
    def environment(self) -> 'TradingEnvironment':
        """A `TradingEnvironment` instance for the agent to trade within."""
        return self._environment

    @environment.setter
    def environment(self, environment: 'TradingEnvironment'):
        self._environment = environment
        # self._environment = DummyVecEnv([lambda: environment])

    def restore_agent(self, path: str):
        """Deserialize the strategy's learning agent from a file.

        Arguments:
            path: The `str` path of the file the agent specification is stored in.
        """
        self._agent = self._model.load(path)

    def save_agent(self, path: str):
        """Serialize the learning agent to a file for restoring later.

        Arguments:
            path: The `str` path of the file to store the agent specification in.
        """
        #os.makedirs(path, exist_ok=True)
        self._agent.save(path)

    def simple_learn(self, total_timesteps=500_000):
        self._agent.learn(total_timesteps=total_timesteps, progress_bar=True)

    def tune(self, steps: int = None, episodes: int = None, callback: Callable[[pd.DataFrame], bool] = None) -> pd.DataFrame:
        raise NotImplementedError

    def run(self, steps: int = None, episodes: int = None, episode_callback: Callable[[pd.DataFrame], bool] = None) -> pd.DataFrame:
        if steps is None and episodes is None:
            raise ValueError(
                'You must set the number of `steps` or `episodes` to run the strategy.')

        steps_completed = 0
        episodes_completed = 0
        average_reward = 0

        obs, info = self._environment.reset()

        performance = {}

        while (steps is not None and (steps == 0 or steps_completed < steps)) or (episodes is not None and episodes_completed < episodes):
            actions, state = self._agent.predict(obs)
            obs, rewards, dones, truncated, info = self._environment.step(actions)

            steps_completed += 1
            if steps_completed == 19:
                debug = 1
            average_reward -= average_reward / steps_completed
            average_reward += rewards/ (steps_completed + 1)

            exchange_performance = info.get('exchange').performance
            performance = exchange_performance if len(exchange_performance) > 0 else performance

            if dones:
                if episode_callback is not None and episode_callback(self._environment._exchange.performance):
                    break
                print("frist done")
                episodes_completed += 1
                obs, info = self._environment.reset()

        print("Finished running strategy.")
        print("Total episodes: {} ({} timesteps).".format(episodes_completed, steps_completed))
        print("Average reward: {}.".format(average_reward))

        return performance

    def backtesting(self):
        performance = {}
        steps_completed = 0
        average_reward = 0
        all_reward = 0
        obs, info = self._environment.reset()
        dones = False
        while not dones:
            actions, state = self._agent.predict(obs)
            obs, rewards, dones, truncated, info = self._environment.step(actions)
            all_reward += rewards
            steps_completed += 1
            average_reward -= average_reward / steps_completed
            average_reward += rewards/ (steps_completed + 1)

            exchange_performance = info.get('exchange').performance
            performance = exchange_performance if len(exchange_performance) > 0 else performance


        print("Finished running strategy.")
        print("Total ({} timesteps).".format(steps_completed))
        print("Average reward: {}.".format(average_reward))
        print("All reward: {}.".format(all_reward))
        return performance