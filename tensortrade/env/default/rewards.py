
from abc import abstractmethod

import numpy as np
import pandas as pd

from tensortrade.env.generic import RewardScheme, TradingEnv
from tensortrade.feed.core import Stream, DataFeed
import math


class TensorTradeRewardScheme(RewardScheme):
    """An abstract base class for reward schemes for the default environment.
    """

    def reward(self, env: 'TradingEnv') -> float:
        return self.get_reward(env.action_scheme.portfolio)

    @abstractmethod
    def get_reward(self, portfolio) -> float:
        """Gets the reward associated with current step of the episode.

        Parameters
        ----------
        portfolio : `Portfolio`
            The portfolio associated with the `TensorTradeActionScheme`.

        Returns
        -------
        float
            The reward for the current step of the episode.
        """
        raise NotImplementedError()


class SimpleProfit(TensorTradeRewardScheme):
    """A simple reward scheme that rewards the agent for incremental increases
    in net worth.

    Parameters
    ----------
    window_size : int
        The size of the look back window for computing the reward.

    Attributes
    ----------
    window_size : int
        The size of the look back window for computing the reward.
    """

    def __init__(self, window_size: int = 1):
        self._window_size = self.default('window_size', window_size)

    def get_reward(self, portfolio: 'Portfolio') -> float:
        """Rewards the agent for incremental increases in net worth over a
        sliding window.

        Parameters
        ----------
        portfolio : `Portfolio`
            The portfolio being used by the environment.

        Returns
        -------
        float
            The cumulative percentage change in net worth over the previous
            `window_size` time steps.
        """
        net_worth = [nw['net_worth'] for nw in portfolio.performance.values()]
        if len(net_worth) > 1:
            return net_worth[-1] / net_worth[-min(len(net_worth), self._window_size + 1)] - 1.0
        else:
            return 0.0

class RiskAdjustedReturns(TensorTradeRewardScheme):
    """A reward scheme that rewards the agent for increasing its net worth,
    while penalizing more volatile strategies.

    Parameters
    ----------
    return_algorithm : {'sharpe', 'sortino'}, Default 'sharpe'.
        The risk-adjusted return metric to use.
    risk_free_rate : float, Default 0.
        The risk free rate of returns to use for calculating metrics.
    target_returns : float, Default 0
        The target returns per period for use in calculating the sortino ratio.
    window_size : int
        The size of the look back window for computing the reward.
    """

    def __init__(self,
                 return_algorithm: str = 'sharpe',
                 risk_free_rate: float = 0.,
                 target_returns: float = 0.,
                 window_size: int = 1) -> None:
        algorithm = self.default('return_algorithm', return_algorithm)

        assert algorithm in ['sharpe', 'sortino']

        if algorithm == 'sharpe':
            return_algorithm = self._sharpe_ratio
        elif algorithm == 'sortino':
            return_algorithm = self._sortino_ratio

        self._return_algorithm = return_algorithm
        self._risk_free_rate = self.default('risk_free_rate', risk_free_rate)
        self._target_returns = self.default('target_returns', target_returns)
        self._window_size = self.default('window_size', window_size)

    def _sharpe_ratio(self, returns: 'pd.Series') -> float:
        """Computes the sharpe ratio for a given series of a returns.

        Parameters
        ----------
        returns : `pd.Series`
            The returns for the `portfolio`.

        Returns
        -------
        float
            The sharpe ratio for the given series of a `returns`.

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Sharpe_ratio
        """
        return (np.mean(returns) - self._risk_free_rate + 1e-9) / (np.std(returns) + 1e-9)

    def _sortino_ratio(self, returns: 'pd.Series') -> float:
        """Computes the sortino ratio for a given series of a returns.

        Parameters
        ----------
        returns : `pd.Series`
            The returns for the `portfolio`.

        Returns
        -------
        float
            The sortino ratio for the given series of a `returns`.

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Sortino_ratio
        """
        downside_returns = returns.copy()
        downside_returns[returns < self._target_returns] = returns ** 2

        expected_return = np.mean(returns)
        downside_std = np.sqrt(np.std(downside_returns))

        return (expected_return - self._risk_free_rate + 1e-9) / (downside_std + 1e-9)

    def get_reward(self, portfolio: 'Portfolio') -> float:
        """Computes the reward corresponding to the selected risk-adjusted return metric.

        Parameters
        ----------
        portfolio : `Portfolio`
            The current portfolio being used by the environment.

        Returns
        -------
        float
            The reward corresponding to the selected risk-adjusted return metric.
        """
        net_worth = [nw['net_worth'] for nw in portfolio.performance.values()][-(self._window_size + 1):]
        returns = pd.Series(net_worth).pct_change().dropna()
        risk_adjusted_return = self._return_algorithm(returns)
        return risk_adjusted_return


class PBR(TensorTradeRewardScheme):
    """A reward scheme for position-based returns.

    * Let :math:`p_t` denote the price at time t.
    * Let :math:`x_t` denote the position at time t.
    * Let :math:`R_t` denote the reward at time t.

    Then the reward is defined as,
    :math:`R_{t} = (p_{t} - p_{t-1}) \cdot x_{t}`.

    Parameters
    ----------
    price : `Stream`
        The price stream to use for computing rewards.
    """

    registered_name = "pbr"

    def __init__(self, price: 'Stream') -> None:
        super().__init__()
        self.position = -1

        r = Stream.sensor(price, lambda p: p.value, dtype="float").diff()
        position = Stream.sensor(self, lambda rs: rs.position, dtype="float")

        reward = (position * r).fillna(0).rename("reward")

        self.feed = DataFeed([reward])
        self.feed.compile()

    def on_action(self, action: int) -> None:
        self.position = -1 if action == 0 else 1

    def get_reward(self, portfolio: 'Portfolio') -> float:
        return self.feed.next()["reward"]

    def reset(self) -> None:
        """Resets the `position` and `feed` of the reward scheme."""
        self.position = -1
        self.feed.reset()


class AnomalousProfit(TensorTradeRewardScheme):
    """A simple reward scheme that rewards the agent for exceeding a 
    precalculated percentage in the net worth.

    Parameters
    ----------
    threshold : float
        The minimum value to exceed in order to get the reward.

    Attributes
    ----------
    threshold : float
        The minimum value to exceed in order to get the reward.
    """

    registered_name = "anomalous"

    def __init__(
        self, 
        threshold: float = 0.02, 
        window_size: int = 1):
        
        self._window_size = self.default('window_size', window_size)
        self._threshold = self.default('threshold', threshold)

    def get_reward(self, portfolio: 'Portfolio') -> float:
        """Rewards the agent for incremental increases in net worth over a
        sliding window.

        Parameters
        ----------
        portfolio : `Portfolio`
            The portfolio being used by the environment.

        Returns
        -------
        int
            Whether the last percent change in net worth exceeds the predefined 
            `threshold`.
        """
        performance = pd.DataFrame.from_dict(portfolio.performance).T
        current_step = performance.shape[0]
        if current_step > 1:
            # Hint: make it cumulative.
            net_worths = performance['net_worth']
            ground_truths = performance * self._threshold
            reward_factor = 2.0 * ground_truths - 1.0
            #return net_worths.iloc[-1] / net_worths.iloc[-min(current_step, self._window_size + 1)] - 1.0
            return (reward_factor * net_worths.abs()).iloc[-1]

        else:
            return 0.0


class PenalizedProfit(TensorTradeRewardScheme):
    """A reward scheme which 
    1. Penalizes net worth loss and decays with the time spent.
    2. Rewards the agent for gaining net worth while holding the asset.

    Parameters
    ----------
    cash_penalty_proportion : float
        cash_penalty_proportion

    Attributes
    ----------
    cash_penalty_proportion : float
        cash_penalty_proportion.
    """

    registered_name = "penalized"

    def __init__(self, cash_penalty_proportion: float = 0.10):
        self._cash_penalty_proportion = \
            self.default(
                'cash_penalty_proportion',
                cash_penalty_proportion)

    def get_reward(self, portfolio: 'Portfolio') -> float:
        """

        Parameters
        ----------
        portfolio : `Portfolio`
            The portfolio being used by the environment.

        Returns
        -------
        int
            A penalized reward.
        """
        performance = pd.DataFrame.from_dict(portfolio.performance).T
        current_step = performance.shape[0]

        if current_step == 0:
            return 0
        
        else:
            #net_worth = [nw['net_worth'] for nw in portfolio.performance.values()]
            #perc_growth = net_worth[-1] / net_worth - 1.0

            #index = self.clock.step
            initial_amount = portfolio.initial_net_worth

            net_worth = performance['net_worth'].iloc[-1] # Last Output
            p_net_worth = performance['net_worth'].iloc[-2] # Previous Output

            #net_worth = performance['net_worth']# Last Output
            #p_net_worth = performance['net_worth'].iloc[-1] # Previous Output

            cash_on_hand = performance['oanda:/USD:/total'].iloc[-1]
            p_cash_on_hand = performance['oanda:/USD:/total'].iloc[-2]

            #cash_on_hand = performance['oanda:/USD:/total']# Last Output
            #p_cash_on_hand = performance['oanda:/USD:/total'].iloc[-1] # Previous Output
            max_cash_limit = net_worth * self._cash_penalty_proportion

            if cash_on_hand > self._cash_penalty_proportion:
                cash_penalty = min(0, (max_cash_limit - cash_on_hand))
            elif cash_on_hand < self._cash_penalty_proportion * net_worth:
                cash_penalty = max(0, (max_cash_limit - cash_on_hand))

            # Logic: 
            # If cash is higher that the penalty limit, the agent is penalized for that by having a -ve step reward
            # If cash is lower than the penalty limit, the agent is rewarded for that by having a +ve step reward
            
            net_worth -= cash_penalty

            reward = (net_worth / p_net_worth) - 1
            # 
            # 
            # 

            reward /= current_step
            '''
            Should the reward be calculated from the last step or from the initial step?

            Two ways to do this:
            1. Compute the cumulated reward from the initial reset
                    The agent might not be motivated to do more.

            2. Compute the reward for each step
                    
            The GOAL for this is to enable the agent to 
            
            Trial 1:
            Objective: Compute the reward based on the difference between the previous and  current step networth to motivate the agent to trade and increase it's rewards by making optimal decisions at each time step

            Hypothesis: The agent may be motivated to take actions that lead to more rewards at each time step. This might lead to the agent exploring optimal actions
            
            Possible downside, the agent may not be able to calculate cumulative rewards at the end of the episode, because the reward is calculated based on the difference between the previous and the current step networth (step-based rewards)

            Outcome: Neutral
            For the first time the reward started in the  positive. 
            However, the rewards for tune and sampler were flat for most of the time. 
            Evaluation rewards were growing.
            The episode reward hist data showed that the rewards remained in the same range for most of the time. Meaning that each time step led to the same reward.

            Why is this not good?
            Because the goal of the agent is to increase the rewards indefinitely leading to the doubling of the account every 2 weeks. 
            
            A possible fix to the downside of this is using a a rolling window of rewards (cumulative rewards) to compute the reward.


            TODO: Modify the reward scheme to:
            1. Reward or penalize for each trade opened and closed
            2. Reward the agent over a rolling window instead of the current step.

            # Trial 2:
            Objective: Compute the reward based on the networth difference between the first and last steps of a rolling window.

            Hypotheis: The agent may not be taking actions at each timestep to maximize the reward. Therefore, cumulating rewards over a rolling window may motivate the agent to take actions that lead to more cumulative rewards at each time step.

            Possible Downside: UNKNOWN

            Outcome: UNKNOWN
            
            '''
            return reward
            

_registry = {
    'simple': SimpleProfit,
    'risk-adjusted': RiskAdjustedReturns,
    'pbr': PBR,
    'Re-Pe': PenalizedProfit,
    'AP': AnomalousProfit
}


def get(identifier: str) -> 'TensorTradeRewardScheme':
    """Gets the `RewardScheme` that matches with the identifier.

    Parameters
    ----------
    identifier : str
        The identifier for the `RewardScheme`

    Returns
    -------
    `TensorTradeRewardScheme`
        The reward scheme associated with the `identifier`.

    Raises
    ------
    KeyError:
        Raised if identifier is not associated with any `RewardScheme`
    """
    if identifier not in _registry.keys():
        msg = f"Identifier {identifier} is not associated with any `RewardScheme`."
        raise KeyError(msg)
    return _registry[identifier]()
