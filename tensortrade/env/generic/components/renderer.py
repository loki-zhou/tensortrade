# Copyright 2020 The TensorTrade Authors.
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
# limitations under the License

from abc import ABCMeta, abstractmethod
from typing import List, Optional

from tensortrade.core import Component
import pandas as pd
from datetime import datetime
from collections import OrderedDict

'''
Notes for upgrading the renderer
https://younis.dev/blog/render-api/
'''

class Renderer(Component, metaclass=ABCMeta):
    """An abstract class of a component for rendering a view of the environment at each step of
    an episode."""

    registered_name = "renderer"

    @abstractmethod
    def render(
        self, 
        env: 'TradingEnv', # type: ignore
        **kwargs):
        """Renders a view of the environment at the current step of an episode.

        Parameters
        ----------
        env: 'TradingEnv'
            The trading environment.
        mode: 'human'
            The render mode
        kwargs : keyword arguments
            Additional keyword arguments for rendering the environment.
        """
        raise NotImplementedError()

    def save(self) -> None:
        """Saves the rendered view of the environment."""
        pass

    def reset(self) -> None:
        """Resets the renderer."""
        pass

    def close(self) -> None:
        """Closes the renderer."""
        pass

class AggregateRenderer(Renderer):
    """A renderer that aggregates compatible renderers so they can all be used
    to render a view of the environment.

    Parameters
    ----------
    renderers : List[Renderer]
        A list of renderers to aggregate.

    Attributes
    ----------
    renderers : List[Renderer]
        A list of renderers to aggregate.
    """

    def __init__(
        self, 
        renderers: List[Renderer],
        ) -> None:
        super().__init__()
        self.renderers = renderers

    def render(self, env: 'TradingEnv', # type: ignore 
               **kwargs) -> None:
        for r in self.renderers:
            r.render(env, **kwargs)

    def save(self) -> None:
        for r in self.renderers:
            r.save()

    def reset(self) -> None:
        for r in self.renderers:
            r.reset()

    def close(self) -> None:
        for r in self.renderers:
            r.close()


class BaseRenderer(Renderer):
    """The abstract base renderer to be subclassed when making a renderer
    the incorporates a `Portfolio`.
    """

    def __init__(
            self,
            render_mode: Optional[str] = None):
        super().__init__()
        self._max_episodes = None
        self._max_steps = None
        self.render_mode =  render_mode

    @staticmethod
    def _create_log_entry(
        episode: int = None,
        max_episodes: int = None,
        step: int = None,
        max_steps: int = None,
        date_format: str = "%Y-%m-%d %H:%M:%S") -> str:
        """
        Creates a log entry to be used by a renderer.

        Parameters
        ----------
        `episode` : int
            The current episode.
        `max_episodes` : int
            The maximum number of episodes that can occur.
        `step` : int
            The current step of the current episode.
        `max_steps` : int
            The maximum number of steps within an episode that can occur.
        `date_format` : str
            The format for logging the date.

        Returns
        -------
        str
            a log entry
        """
        log_entry = f"[{datetime.now().strftime(date_format)}]"

        if episode is not None:
            log_entry += f" Episode: {episode + 1}/{max_episodes if max_episodes else ''}"

        if step is not None:
            log_entry += f" Step: {step}/{max_steps if max_steps else ''}"

        return log_entry

    def render(self, 
            env: 'TradingEnv', #type: ignore
            **kwargs):
        if self.render_mode is not None: 
            price_history = None
            if len(env.observer.renderer_history) > 0:
                price_history = pd.DataFrame(env.observer.renderer_history)

            performance = pd.DataFrame.from_dict(env.action_scheme.portfolio.performance, orient='index')

            self.render_env(
                episode=kwargs.get("episode", None),
                max_episodes=kwargs.get("max_episodes", None),
                step=env.clock.step,
                max_steps=kwargs.get("max_steps", None),
                price_history=price_history,
                net_worth=performance.net_worth,
                performance=performance.drop(columns=['base_symbol']),
                trades=env.action_scheme.broker.trades
            )
        else:
            pass

    @abstractmethod
    def render_env(
        self,
        episode: int = None,
        max_episodes: int = None,
        step: int = None,
        max_steps: int = None,
        price_history: 'pd.DataFrame' = None,
        net_worth: 'pd.Series' = None,
        performance: 'pd.DataFrame' = None,
        trades: 'OrderedDict' = None) -> None:
        """Renderers the current state of the environment.

        Parameters
        ----------
        episode : int
            The episode that the environment is being rendered for.
        max_episodes : int
            The maximum number of episodes that will occur.
        step : int
            The step of the current episode that is happening.
        max_steps : int
            The maximum number of steps that will occur in an episode.
        price_history : `pd.DataFrame`
            The history of instrument involved with the environment. The
            required columns are: date, open, high, low, close, and volume.
        net_worth : `pd.Series`
            The history of the net worth of the `portfolio`.
        performance : `pd.Series`
            The history of performance of the `portfolio`.
        trades : `OrderedDict`
            The history of trades for the current episode.
        """
        raise NotImplementedError()

    def save(self) -> None:
        """Saves the rendering of the `TradingEnv`.
        """
        pass

    def reset(self) -> None:
        """Resets the renderer.
        """
        pass
