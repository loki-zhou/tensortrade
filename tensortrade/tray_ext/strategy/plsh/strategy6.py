
# base class
# RL Agent
from ray.rllib.algorithms import ppo

# Call backs
from tensortrade.tray_ext.rayExtension.callbacks.recordShortNetWorthCallback import (
    RecordNetWorthCallback,
)
from tensortrade.tray_ext.strategies.base_strategy import Strategy

# Environment
from tensortrade.tray_ext.strategy.plsh.environment6 import create_env


class PPO_Sinewave_PLSH_SNC(Strategy):

    '''
    Strategy 6:
      DRL: 
        PPO with a custom MLP architecture with default attension config
      Environment: sinewave
        Data: Generated Sinewave
          Training: 5 peaks - Peaks is not an attribute of sinewave generator.
          Evaluation: 2 peaks
          Testing: 3 peaks
        Observation Space: 
          price -> close,
          price -> rolling mean (10 data points),
          price -> rolling mean (20 data points),
          price -> rolliong mean (30 data points),
          price -> log difference
      Action Space: proportion-buy-sell-short-hold
      Reward Strategy: short-networth-change
    '''
    def __init__(self):
      # run configuration
      self.max_epoch = 50
      self.net_worth_threshold = 1000
      self.patience = 1
      self.evaluation_frequency = 1
      self.log_name = "sinewave/strategy6"
      self.log_dir = "C:/$Quant Connect Lean Engine/Projects/Research/trials_3"

      # configure the train environment
      self.env_train_config = {
        "type": "train",
        "window_size": 100,
        "min_periods": 30,
        "max_allowed_loss": 1, # allow for 100% loss of funds
        "period": 50, # the number of periods to generate with the sine wave
        "render_env": True,
        "trading_days": 1000,
        "log_name": self.log_name,
        "log_dir": self.log_dir,
        "action_scheme": 'proportion-buy-sell-short-hold',
        "reward_scheme": 'short-networth-change'
      }
      # Configure the algorithm.
      self.config = {
        "env": "TradingEnv",
        "env_config": self.env_train_config,  # config to pass to env class
        "evaluation_interval": self.evaluation_frequency,
        "evaluation_duration_unit": 'episodes',
        "evaluation_num_workers": 1,
        "evaluation_config": {
            "env_config": self.env_train_config,
            "render_env": True,
            "explore": False,
        },
        "num_workers": 2,
        "batch_mode": "complete_episodes",
        "callbacks": RecordNetWorthCallback,
        "framework": "tf2",
        "eager_tracing": True,
        # "horizon": 30 # Deprecated
      }
      ppo_config = ppo.DEFAULT_CONFIG.copy()
      ppo_config.update(self.config)
      self.config = ppo_config
      self.algorithm_name="PPO"
      self.create_env = create_env
      self.agent = ppo.PPO
'''
def main():
  strategy = PPO_Sinewave_PLSH_SNC()
  strategy.clearLogs()
  strategy.train()

main()
'''