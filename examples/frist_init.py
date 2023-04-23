import pandas as pd
from tensortrade.exchanges.simulated import SimulatedExchange
from tensortrade.features import FeaturePipeline
from tensortrade.features.scalers import MinMaxNormalizer
from tensortrade.features.stationarity import FractionalDifference
from tensortrade.features.indicators import SimpleMovingAverage
from tensortrade.rewards import SimpleProfit, RiskAdjustedReturns
from tensortrade.actions import DiscreteActions
from gymnasium.utils.env_checker import check_env
from tensortrade.strategies import StableBaselinesTradingStrategy
from tensortrade.environments import TradingEnvironment


df = pd.read_csv('data/Coinbase_BTCUSD_1h.csv', skiprows=1)
exchange = SimulatedExchange(data_frame=df, base_instrument='USD', pretransform=True, window_size=20)
# exchange = SimulatedExchange(data_frame=df, base_instrument='USD', pretransform=True)

normalize_price = MinMaxNormalizer(["open", "high", "low", "close"], inplace=False)
difference_all = FractionalDifference(["open", "high", "low", "close"], difference_order=0.6, inplace=False)
moving_averages = SimpleMovingAverage(["open", "high", "low", "close"], inplace=False)
feature_pipeline = FeaturePipeline(steps=[normalize_price, moving_averages, difference_all])

# normalize_price = MinMaxNormalizer(["open", "high", "low", "close"])
# difference_all = FractionalDifference(["open", "high", "low", "close"], difference_order=0.6)
# feature_pipeline = FeaturePipeline(steps=[normalize_price,  difference_all])

#exchange.feature_pipeline = feature_pipeline




action_scheme = DiscreteActions(n_actions=20, instrument='BTC')

# reward_scheme = SimpleProfit()
reward_scheme = RiskAdjustedReturns()


from tensortrade.environments import TradingEnvironment

environment = TradingEnvironment(exchange=exchange,
                                 feature_pipeline=feature_pipeline,
                                 action_scheme=action_scheme,
                                 reward_scheme=reward_scheme)

from stable_baselines3 import PPO

model = PPO
policy = "MlpPolicy"
params = { "learning_rate": 1e-5, 'batch_size': 64, 'verbose': 1 }



strategy = StableBaselinesTradingStrategy(environment=environment,
                                          model=model,
                                          policy=policy,
                                          model_kwargs=params)
if 1:
    strategy.simple_learn(total_timesteps=500_000)
    strategy.save_agent(path="agents/ppo_btc_1h")
    # strategy.run(steps=10000)
else:
    strategy.restore_agent(path="agents/ppo_btc_1h")
    performance = strategy.backtesting()
    performance.net_worth.plot()


# performance = strategy.run(steps=10000, episodes=1)
# performance.net_worth.plot()