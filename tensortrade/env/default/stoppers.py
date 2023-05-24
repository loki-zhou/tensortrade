
from tensortrade.env.generic import TradingEnv
from  tensortrade.env.generic.components.stopper import Stopper as Terminate
from ray.tune import Stopper
import time

class MaxLossStopper(Stopper): # Will be deprecated
    """A stopper that stops an episode if the portfolio has lost a particular
    percentage of its wealth.

    Parameters
    ----------
    max_allowed_loss : float
        The maximum percentage of initial funds that is willing to
        be lost before stopping the episode.

    Attributes
    ----------
    max_allowed_loss : float
        The maximum percentage of initial funds that is willing to
        be lost before stopping the episode.

    Notes
    -----
    This stopper also stops if it has reached the end of the observation feed.
    """

    def __init__(self, max_allowed_loss: float):
        super().__init__()
        self.max_allowed_loss = max_allowed_loss

    def stop(self, env: 'TradingEnv') -> bool:
        c1 = env.action_scheme.portfolio.profit_loss > self.max_allowed_loss
        c2 = not env.observer.has_next()
        return c1 or c2

class NetWorthstopper(Stopper, Terminate):
    """
    A stopper that stops an episode if the portfolio has lost a particular
    percentage of its wealth or has reached it's objective.

    Parameters
    ----------
    max_allowed_loss : float
        The maximum percentage of initial funds that is willing to
        be lost before stopping the episode.

    Attributes
    ----------
    max_allowed_loss : float
        The maximum percentage of initial funds that is willing to
        be lost before stopping the episode.

    Notes
    -----
    This stopper also stops if it has reached the end of the observation feed.
    """
    def __init__(self, 
        net_worth_mean: int,
        max_allowed_loss: float,
        patience: int = 0,
        
    ):
        super().__init__()
        self.max_allowed_loss = max_allowed_loss
        self._start = time.time()
        self._net_worth_mean = net_worth_mean
        self._patience = patience
        self._iterations = 0
        self.stop = False
        return

    def has_reached_objective(self, result):
        if "net_worth_max" not in result["custom_metrics"]:
            self.stop = False
        elif result["custom_metrics"]["net_worth_mean"] >= self._net_worth_mean:
            self.stop = True
        return self.stop

    def __call__(self, trial_id, result):
        if self.has_reached_objective(result):
            self._iterations +=1
        else:
            self._iterations = 0
        return self.stop_all()

    def stop_all(self):
        return self.stop and self._iterations >= self._patience
    
    def stop(self, env: 'TradingEnv') -> bool:
        c1 = env.action_scheme.portfolio.profit_loss > self.max_allowed_loss
        c2 = not env.observer.has_next()
        return c1 or c2

'''
class ShortStopper(Terminate):
    """A stopper that stops an episode if the agents deposit margin goes below the maintenance margin limit

    This stopper assumes that only short positions have margins which is false in the real world.

    Attributes
    ----------
    max_allowed_loss : float
        The maximum percentage of initial funds that is willing to
        be lost before stopping the episode.

    Notes
    -----
    This stopper also stops if it has reached the end of the observation feed.
    """

    def __init__(self):
        super().__init__()

    def borrow_limit_reached(self, env: 'TradingEnv'):
        action_scheme = env.action_scheme
        maintenance_margin = action_scheme.maintenance_margin
        deposit_margin = action_scheme.deposit_margin.total_balance.as_float()
        current_short_value = action_scheme.borrow_asset.convert(action_scheme.exchange_pair)
        margin_requirement = (1 + maintenance_margin)
        if current_short_value > 0:
            margin_threshold = margin_requirement * current_short_value
            if deposit_margin < margin_threshold:
                return True
        return False

    def cash_minimum_hit(self, env: 'TradingEnv'):
        action_scheme: PLSH = env.action_scheme
        return action_scheme.cash.balance.as_float() < action_scheme.minimum_short_deposit

    def terminate(self, env: 'TradingEnv') -> bool:
        c1 = self.borrow_limit_reached(env)
        # c2 = self.cash_minimum_hit(env)
        stop = False
        if c1:
            stop = True
        return stop


class LongShortStopper(Terminate):
    """
    Work in progress
    A stopper that stops an episode if the agents receives a margin call

    The goal of this stopper is to emulate the real world events surrounding margin calls.

    Attributes
    ---------- 
    margin_call_level : float
        The maximum percentage of initial funds that is willing to
        be lost before stopping the episode.
    TODO: Implement margin call attributes

    Notes
    -----
    This stopper also stops if it has reached the end of the observation feed.
    TODO: Implement margin call logic.
    TODO: Implement cash minimum hit logic.
    TODO: Implement margin wallets:
            The margin wallets are used to track the margin requirements for the assets.
            An easier way to implement this is to use the `MarginWallet` class which locks this wallet so that it's not usable for trading.
            This way, funds can be transfered from the margin wallet to the cash wallet and back. 
            The locked amounts are determined by maintenance margin which changes based on
    """

    def __init__(self):
        super().__init__()

    def margin_limit_reached(self, env: 'TradingEnv'):
        action_scheme = env.action_scheme
        maintenance_margin = action_scheme.maintenance_margin
        initial_deposit_margin = action_scheme.initial_deposit_margin.total_balance.as_float()
        margin_requirement = (1 + maintenance_margin)

        # Short positions
        current_short_value = action_scheme.borrow_asset.convert(action_scheme.exchange_pair)
        if current_short_value > 0:
            margin_threshold = margin_requirement * current_short_value
            if initial_deposit_margin < margin_threshold:
                return True

        # New mods
        margin_limit = initial_deposit_margin * margin_requirement - initial_deposit_margin
        margin_balance = action_scheme.margin_wallet.total_balance.as_float()
        current_long_value = action_scheme.long_asset.convert(action_scheme.exchange_pair)
        if current_long_value > 0:
            margin_threshold = margin_requirement * current_long_value
            if initial_deposit_margin < margin_threshold:
                return True
        return False

    def cash_minimum_hit(self, env: 'TradingEnv'):
        action_scheme: PLSH = env.action_scheme
        return action_scheme.cash.balance.as_float() < action_scheme.minimum_short_deposit

    def terminate(self, env: 'TradingEnv') -> bool:
        c1 = self.margin_limit_reached(env)
        # c2 = self.cash_minimum_hit(env)
        stop = False
        if c1:
            stop = True
        return stop
'''