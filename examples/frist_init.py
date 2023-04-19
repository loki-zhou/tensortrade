import pandas as pd
from tensortrade.exchanges.simulated import SimulatedExchange
from tensortrade.actions import DiscreteActions
from tensortrade.features import FeaturePipeline
from tensortrade.features.scalers import MinMaxNormalizer
from tensortrade.features.stationarity import FractionalDifference
from tensortrade.rewards import SimpleProfit

from gymnasium.utils.env_checker import check_env

df = pd.read_csv('data/Coinbase_BTCUSD_1h.csv', skiprows=1)
exchange = SimulatedExchange(data_frame=df, base_instrument='USD', pretransform=True)



normalize_price = MinMaxNormalizer(["open", "high", "low", "close"])
difference_all = FractionalDifference(["open", "high", "low", "close"], difference_order=0.6)

feature_pipeline = FeaturePipeline(steps=[normalize_price, difference_all])

exchange.feature_pipeline = feature_pipeline


action_scheme = DiscreteActions(n_actions=20, instrument='BTC')
reward_scheme = SimpleProfit()
from tensortrade.environments import TradingEnvironment

environment = TradingEnvironment(exchange=exchange,
                                 feature_pipeline=feature_pipeline,
                                 action_scheme=action_scheme,
                                 reward_scheme=reward_scheme)


check_env(environment)