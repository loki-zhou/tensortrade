from typing import Tuple
from tensortrade.env.default.rewards import TensorTradeRewardScheme
from tensortrade.feed.core import Stream, DataFeed

from tensortrade.env.generic import TradingEnv
from tensortrade.oms.wallets import Portfolio
from tensortrade.TT_extension.actions import PLSH
import numpy as np 

class PPBR_Custom(TensorTradeRewardScheme):
    """
    Work in Progress
    This is a reward scheme that is currently built on the following premises:

        1. Reward the agent for profitable trades, whether long or short trades
        2. Rewards the agent for increasing the value of the portfolio, regardless of whether it is a long or short trade.
        3. 
    Future Mods:
        Penalties
        1. Penalizes the agent for entering trades that lead to the account balance falling. This can be implemented through a account loss penalty
        2. Penalizes the agent open positions are closed with under expectation or low %profit. This can be implemented through a low profit penalty
        3. Penalize the agent for not maintaining a healthy account balance. This can be implemented through a account balance penalty.
        4. Penalizing the agent for breaching the margin call limits

        Rewards:
        1. Reward the agent for holding a profitable position for a certain amount of time and exiting a the profitable trade at the optimal time. This is to ensure that the agent doesn't leave any money on the table. 
        2. Reward the agent for exiting losing trades at the best time. A stop loss can be implemented but is not optimal for this reward scheme.
        3. Reward the agent for maintaining a healthy account balance.
        
        These mods will enable the model to trade with high confidence and manage cash reserves.

        Reward at any step can be given as follows
            r_i = (sum(cash, asset_value) + additional_reward - total_penalty - initial_cash) / initial_cash / days_elapsed
            , where total_penalty = cash_penalty + stop_loss_penalty + low_profit_penalty
                    cash_penalty = max(0, sum(cash, asset_value)*cash_penalty_proportion-cash)
                    account_loss_penalty = -1 * dot(holdings,negative_closing_diff_avg_buy)
                    low_profit_penalty = -1 * dot(holdings,negative_profit_sell_diff_avg_buy)
                    additional_reward = dot(holdings,positive_profit_sell_diff_avg_buy)

        Such a reward function would take into account a profit/loss ratio constraint, liquidity requirement, as well as long-term accrued rewards.
        
    
    Future Mods 2:
    1. Include the following for the agent to consider while trading:
        1.1  A profit_loss_ratio - Expected profit/loss ratio.
        1.2 Patience parameter - An option to choose whether end the cycle when running out of cash or just don't trade until cash is deposited.
                A penalty for breaching this will be implemented but margin trading is going to be a complex issue.
    

    A reward function with the above mods would force the model to trade only when it's really confident to do so.

    """

    registered_name = "cpbr"

    def __init__(self,
        price: 'Stream',
        starting_value: float) -> None:
        super().__init__()
        self.position = -1
        self.proportion = 0
        
        # New Mods:
        self.net_worth_history = []
        self.starting_value = starting_value
        self.previous_net_worth = starting_value

        position_difference = Stream.sensor(price, lambda p: p.value, dtype="float").diff()
        direction = Stream.sensor(self, lambda self: self.position, dtype="float")

        reward = (direction * position_difference).fillna(0).rename("reward")
        self.feed = DataFeed([reward])
        self.feed.compile()

    def on_action(self, actions: Tuple) -> None:
        self.proportion = actions[1][0]

    def get_reward(self, portfolio: 'Portfolio', action_scheme: PLSH) -> float:
        """
        Parameters
        ----------
        short_action_scheme :
            The action scheme used for managing simple short positions.
        Returns
        -------
        float
            The difference in networth as profit / loss 
        """
        '''
        The code below is calculating rewards for a trading agent. The first part of the code calculates the reward based on the position of the agent, and assigns it to the variable 'proportion_reward'. The second part of the code calculates the reward based on the networth of the agent, and assigns it to 'networth_reward'. It then stores the current networth in 'previous_net_worth' and adds it to a list called 'net_worth_history'. If this list is longer than 100 elements, it will calculate an average of all elements in this list and assign it to 'networth_reward'. Finally, it returns both rewards added together.
        '''
        # Position Based Rewards
        reward = self.feed.next()["reward"]
        reward_ = reward / 100
        proportion_ =  self.proportion / 100
        proportion_reward = reward * proportion_

        # Networth Based Rewards
        asset_balance = action_scheme.asset.balance.convert(action_scheme.exchange_pair)
        cash_balance = action_scheme.cash_balance
        networth = (cash_balance + asset_balance).as_float()
        networth_change = networth - self.previous_net_worth
        networth_perc_change = networth_change / self.previous_net_worth * 100.0

        networth_reward = networth_perc_change  * proportion_ * reward_ * -1.0

        self.previous_net_worth = networth
        self.net_worth_history.append(networth)

        if len(self.net_worth_history) > 100:
            self.net_worth_history.pop(0)
            networth_reward = np.mean(self.net_worth_history)
            self.net_worth_history = []


        # Penalties

        return proportion_reward + networth_reward


    def reset(self) -> None:
        """Resets the history and previous net worth of the reward scheme."""
        self.position = -1
        self.feed.reset()
        self.proportion = 0
        


class PPBR_Original(TensorTradeRewardScheme):
    """A simple reward scheme that rewards the agent based on the change in its networth
    """

    registered_name = "cpbr"

    def __init__(self, price: 'Stream') -> None:
        super().__init__()
        self.position = -1
        self.proportion = 0

        position_difference = Stream.sensor(price, lambda p: p.value, dtype="float").diff()
        direction = Stream.sensor(self, lambda self: self.position, dtype="float")

        reward = (direction * position_difference).fillna(0).rename("reward")
        self.feed = DataFeed([reward])
        self.feed.compile()

    def on_action(self, actions: Tuple) -> None:
        self.proportion = actions[1][0]

    def get_reward(self, portfolio: 'Portfolio') -> float:
        """
        Parameters
        ----------
        short_action_scheme : `SH`
            The action scheme used for managing simple short positions.
        Returns
        -------
        float
            The difference in networth as profit / loss 
        """
        reward = self.feed.next()["reward"]
        preportion_reward = reward * self.proportion / 100

        return preportion_reward

    def reset(self) -> None:
        """Resets the history and previous net worth of the reward scheme."""
        self.position = -1
        self.feed.reset()
        self.proportion = 0
        


class NWC(TensorTradeRewardScheme):
    """
        A simple reward scheme that rewards the agent based on the 
        change in its networth

        # Built for the Proportional Long Short Hold Actions
    """

    registered_name = "PPBR"

    def __init__(self, starting_value: float):
        self.net_worth_history = []
        self.starting_value = starting_value
        self.previous_net_worth = starting_value

    def reward(self, env: TradingEnv) -> float:
        return self.get_reward(env.action_scheme)

    def get_reward(self, action_scheme: PLSH) -> float:
        """
        Parameters
        ----------
        action_scheme : `PLSH`
            The action scheme used for managing complex buy sell hold.
        Returns
        -------
        float
            The difference in networth as profit / loss 
        """
        asset_balance = action_scheme.asset.balance.convert(action_scheme.exchange_pair)
        cash_balance = action_scheme.cash.balance
        net_worth = (asset_balance + cash_balance).as_float()
        net_worth_change = net_worth - self.previous_net_worth
        self.previous_net_worth = net_worth
        self.net_worth_history.append(net_worth)

        return net_worth_change

    def reset(self) -> None:
        """Resets the history and previous net worth of the reward scheme."""
        self.net_worth_history = []
        self.previous_net_worth = self.starting_value

# TODO: Merge the two if possible


class SNWC(TensorTradeRewardScheme):
    """A simple reward scheme that rewards the agent based on the change in its networth
    """

    def __init__(self, starting_value: float):
        self.net_worth_history = []
        self.starting_value = starting_value
        self.previous_net_worth = starting_value

    def reward(self, env: TradingEnv) -> float:
        return self.get_reward(env.action_scheme)

    def get_reward(self, action_scheme: PLSH) -> float:
        """
        Parameters
        ----------
        short_action_scheme : `SH`
            The action scheme used for managing simple short positions.
        Returns
        -------
        float
            The difference in networth as profit / loss 
        """
        asset_balance = action_scheme.asset.balance.convert(action_scheme.exchange_pair)
        cash_balance = action_scheme.cash.balance
        deposit_margin = action_scheme.deposit_margin.balance
        borrowed_cash = action_scheme.borrow_asset.convert(action_scheme.exchange_pair)
        net_worth = (asset_balance + cash_balance + deposit_margin - borrowed_cash).as_float()
        net_worth_change = net_worth - self.previous_net_worth
        self.previous_net_worth = net_worth
        self.net_worth_history.append(net_worth)

        return net_worth_change

    def reset(self) -> None:
        """Resets the history and previous net worth of the reward scheme."""
        self.net_worth_history = []
        self.previous_net_worth = self.starting_value
 