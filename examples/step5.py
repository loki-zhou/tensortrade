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
from ray.tune.schedulers import PopulationBasedTraining


def explore(config):
    # ensure we collect enough timesteps to do sgd
    if config["train_batch_size"] < config["sgd_minibatch_size"] * 2:
        config["train_batch_size"] = config["sgd_minibatch_size"] * 2
    # ensure we run at least one sgd iter
    if config["num_sgd_iter"] < 1:
        config["num_sgd_iter"] = 1
    return config

import random
hyperparam_mutations = {
    "lambda": lambda: random.uniform(0.9, 1.0),
    "clip_param": lambda: random.uniform(0.01, 0.5),
    "lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
    "num_sgd_iter": lambda: random.randint(1, 30),
    "sgd_minibatch_size": lambda: random.randint(128, 16384),
    "train_batch_size": lambda: random.randint(2000, 160000),
}
pbt = PopulationBasedTraining(
    time_attr="time_total_s",
    perturbation_interval=120,
    resample_probability=0.25,
    # Specifies the mutations of these hyperparams
    hyperparam_mutations=hyperparam_mutations,
    custom_explore_fn=explore,
)
stopping_criteria = {"training_iteration": 100, "episode_reward_mean": 600}

config={
    "env": "TradingEnv",
    "env_config": {
        "window_size": 25
    },
    "log_level": "DEBUG",
    "framework": "torch",
    "ignore_worker_failures": True,
    "num_workers": 1,
    "num_gpus": 0,
    "clip_rewards": True,
    "lr": 1e-4,
    "gamma": 0,
    "observation_filter": "MeanStdFilter",
    "vf_loss_coeff": 0.5,
    "entropy_coeff": 0.01,
    "model": {"free_log_std": True},
    "lambda": 0.95,
    "clip_param": 0.2,
    "num_sgd_iter": tune.choice([10, 20, 30]),
    "sgd_minibatch_size": tune.choice([128, 512, 2048]),
    "train_batch_size": tune.choice([10000, 20000, 40000]),
}
stop = {
    "episode_reward_mean": 990,
}

ray.init()
tuner  = tune.Tuner(
    "PPO", param_space=config,
    tune_config=tune.TuneConfig(
        metric="episode_reward_mean",
        mode="max",
        scheduler=pbt,
        num_samples=1
    ),
    run_config=air.RunConfig(stop=stopping_criteria, checkpoint_config=air.CheckpointConfig(checkpoint_at_end=True))
)
results = tuner.fit()

# Get the best result based on a particular metric.
# best_result = results.get_best_result(metric="episode_reward_mean", mode="max")
best_result = results.get_best_result()
print("Best performing trial's final set of hyperparameters:\n")
pprint(
    {k: v for k, v in best_result.config.items() if k in hyperparam_mutations}
)
print("\nBest performing trial's final reported metrics:\n")

metrics_to_print = [
    "episode_reward_mean",
    "episode_reward_max",
    "episode_reward_min",
    "episode_len_mean",
]
pprint({k: v for k, v in best_result.metrics.items() if k in metrics_to_print})


# Get the best checkpoint corresponding to the best result.
best_checkpoint = best_result.checkpoint
pprint(best_checkpoint)
#Checkpoint(local_path=C:\Users\loki_\ray_results\PPO\PPO_TradingEnv_dbb99_00000_0_2023-03-16_20-34-13\checkpoint_000006)


import ray.rllib.agents.ppo as ppo
# agent = ppo.PPOTrainer(
#     env="TradingEnv",
#     config=config
# )

#checkpoint_path = r"C:\Users\zhouyi\ray_results\PPO\PPO_TradingEnv_884d9_00000_0_2023-03-17_13-21-12\checkpoint_000030"
# agent.restore(checkpoint_path)
# agent
# best_checkpoint
from ray.rllib.algorithms.algorithm import Algorithm
agent = Algorithm.from_checkpoint(best_result.checkpoint)
#loaded_policy = agent.get_policy()
#agent = Algorithm.from_checkpoint(best_checkpoint)
#%%
# Instantiate the environment
env = create_env({
    "window_size": 25
})

# Run until episode ends
episode_reward = 0
done = False
obs, info = env.reset()
step = 0
while not done:
    action = agent.compute_single_action(obs)
    #action = agent.compute_action(obs)
    next_obs, reward, done, truncated, _  = env.step(action)
    print("action = ", action, "reward = ", reward, " step =", step )
    episode_reward += reward
    step += 1
    obs = next_obs
print("episode_reward = ", episode_reward)
env.render()
