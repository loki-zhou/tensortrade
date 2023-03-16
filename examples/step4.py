import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensortrade.env.generic import Renderer


class PositionChangeChart(Renderer):

    def __init__(self, color: str = "orange"):
        self.color = "orange"

    def render(self, env, **kwargs):
        history = pd.DataFrame(env.observer.renderer_history)

        actions = list(history.action)
        p = list(history.price)

        buy = {}
        sell = {}

        for i in range(len(actions) - 1):
            a1 = actions[i]
            a2 = actions[i + 1]

            if a1 != a2:
                if a1 == 0 and a2 == 1:
                    buy[i] = p[i]
                else:
                    sell[i] = p[i]

        buy = pd.Series(buy)
        sell = pd.Series(sell)

        fig, axs = plt.subplots(1, 2, figsize=(15, 5))

        fig.suptitle("Performance")

        axs[0].plot(np.arange(len(p)), p, label="price", color=self.color)
        axs[0].scatter(buy.index, buy.values, marker="^", color="green")
        axs[0].scatter(sell.index, sell.values, marker="^", color="red")
        axs[0].set_title("Trading Chart")

        performance_df = pd.DataFrame().from_dict(env.action_scheme.portfolio.performance, orient='index')
        performance_df.plot(ax=axs[1])
        axs[1].set_title("Net Worth")

        plt.show()

import ray
import numpy as np
import pandas as pd

from ray import tune, air
from ray.tune.registry import register_env

import tensortrade.env.default as default

from tensortrade.feed.core import DataFeed, Stream
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.wallets import Wallet, Portfolio
from tensortrade.oms.instruments import Instrument
from tensortrade.env.default.rewards import (
    TensorTradeRewardScheme,
    SimpleProfit,
    RiskAdjustedReturns,
    PBR,
)
from tensortrade.env.default.actions import (
    BSH
)

USD = Instrument("USD", 2, "U.S. Dollar")
TTC = Instrument("TTC", 8, "TensorTrade Coin")


def create_env(config):
    x = np.arange(0, 2*np.pi, 2*np.pi / 1001)
    y = 50*np.sin(3*x) + 100

    np.arange(0, 2 * np.pi, 2 * np.pi / 1000)
    p = Stream.source(y, dtype="float").rename("USD-TTC")

    bitfinex = Exchange("bitfinex", service=execute_order)(
        p
    )

    cash = Wallet(bitfinex, 100000 * USD)
    asset = Wallet(bitfinex, 0 * TTC)

    portfolio = Portfolio(USD, [
        cash,
        asset
    ])

    feed = DataFeed([
        p,
        p.rolling(window=10).mean().rename("fast"),
        p.rolling(window=50).mean().rename("medium"),
        p.rolling(window=100).mean().rename("slow"),
        p.log().diff().fillna(0).rename("lr")
    ])

    reward_scheme = PBR(price=p)

    action_scheme = BSH(
        cash=cash,
        asset=asset
    ).attach(reward_scheme)

    renderer_feed = DataFeed([
        Stream.source(y, dtype="float").rename("price"),
        Stream.sensor(action_scheme, lambda s: s.action, dtype="float").rename("action")
    ])

    environment = default.create(
        feed=feed,
        portfolio=portfolio,
        action_scheme=action_scheme,
        reward_scheme=reward_scheme,
        renderer_feed=renderer_feed,
        renderer=PositionChangeChart(),
        window_size=config["window_size"],
        max_allowed_loss=0.6
    )
    return environment

register_env("TradingEnv", create_env)

##############################################################
from ray.rllib.algorithms.ppo import PPOConfig
from pprint import pprint
# config = PPOConfig()
# pprint(config.to_dict())
config={
    "env": "TradingEnv",
    "env_config": {
        "window_size": 25
    },
    "log_level": "DEBUG",
    "framework": "torch",
    "ignore_worker_failures": True,
    "num_workers": 1,
    "num_gpus": 1,
    "clip_rewards": True,
    "lr": 8e-6,
    "lr_schedule": [
        [0, 1e-1],
        [int(1e2), 1e-2],
        [int(1e3), 1e-3],
        [int(1e4), 1e-4],
        [int(1e5), 1e-5],
        [int(1e6), 1e-6],
        [int(1e7), 1e-7]
    ],
    "gamma": 0,
    "observation_filter": "MeanStdFilter",
    "lambda": 0.72,
    "vf_loss_coeff": 0.5,
    "entropy_coeff": 0.01
}
stop = {
    "episode_reward_mean": 700,
}
if 1:
    analysis = tune.Tuner(
        "PPO", param_space=config, run_config=air.RunConfig(stop=stop, checkpoint_config=air.CheckpointConfig(checkpoint_at_end=True))
    ).fit()

    # Get the best result based on a particular metric.
    best_result = analysis.get_best_result(metric="episode_reward_mean", mode="max")

    # Get the best checkpoint corresponding to the best result.
    best_checkpoint = best_result.checkpoint
    pprint(best_checkpoint)
    #Checkpoint(local_path=C:\Users\loki_\ray_results\PPO\PPO_TradingEnv_dbb99_00000_0_2023-03-16_20-34-13\checkpoint_000006)

else:
    import ray.rllib.agents.ppo as ppo
    agent = ppo.PPOTrainer(
        env="TradingEnv",
        config={
            "env_config": {
                "window_size": 25
            },
            "framework": "torch",
            "log_level": "DEBUG",
            "ignore_worker_failures": True,
            "num_workers": 1,
            "num_gpus": 0,
            "clip_rewards": True,
            "lr": 8e-6,
            "lr_schedule": [
                [0, 1e-1],
                [int(1e2), 1e-2],
                [int(1e3), 1e-3],
                [int(1e4), 1e-4],
                [int(1e5), 1e-5],
                [int(1e6), 1e-6],
                [int(1e7), 1e-7]
            ],
            "gamma": 0,
            "observation_filter": "MeanStdFilter",
            "lambda": 0.72,
            "vf_loss_coeff": 0.5,
            "entropy_coeff": 0.01
        }
    )

    checkpoint_path = r"C:\Users\loki_\ray_results\PPO\PPO_TradingEnv_349e6_00000_0_2023-03-16_22-59-52\checkpoint_000010"
    agent.restore(checkpoint_path)
    #%%
    # Instantiate the environment
    env = create_env({
        "window_size": 25
    })

    # Run until episode ends
    episode_reward = 0
    done = False
    obs = env.reset()

    while not done:
        action = agent.compute_single_action(obs)
        obs, reward, done, _, info = env.step(action)
        print("reward", reward)
        episode_reward += reward
    print("episode_reward = ", episode_reward)
    env.render()
