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

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO, DQN
import os
import optuna
from common import SaveOnBestTrainingRewardCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
#verbose 0不打印任何训练信息，1打印训练信息，2打印调试信息
monitor_dir = r'./monitor_log/'
os.makedirs(monitor_dir,exist_ok=True)


def optimize_ppo2(trial):
    return {
        'n_steps': int(trial.suggest_loguniform('n_steps', 16, 2048)),
        'gamma': trial.suggest_loguniform('gamma', 0.9, 0.9999),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1.),
        'ent_coef': trial.suggest_loguniform('ent_coef', 1e-8, 1e-1),
        'clip_range': trial.suggest_uniform('clip_range', 0.1, 0.4),
        'gae_lambda': trial.suggest_uniform('gae_lambda', 0.8, 0.99)
    }

def optimize_agent(trial):
    try:
        env = create_env({
            "window_size": 25
        })
        #env = Monitor(env, monitor_dir)
        model_params = optimize_ppo2(trial)
        model = PPO("MlpPolicy", env, verbose=1,  tensorboard_log="./tensorboard", **model_params)
        model.learn(total_timesteps=60_000)
        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=1)
        env.close()
        SAVE_PATH = os.path.join(monitor_dir, 'trial_{}_best_model'.format(trial.number))
        model.save(SAVE_PATH)
        return mean_reward
    except Exception as e:
        print(e)
        return -1000

def optimize():
    study = optuna.create_study(direction='maximize')
    study.optimize(optimize_agent, n_trials=20,n_jobs=2)
    print('Number of finished trials: ', len(study.trials))

    print('Best trial:')
    trial = study.best_trial

    print('Value: ', trial.value)

    print('Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))

    return study.trials_dataframe()

# model = PPO("MlpPolicy", env, verbose=1,
#             gamma = 0,
#             learning_rate=8e-6,
#             ent_coef = 0.01)

# callback = SaveOnBestTrainingRewardCallback(check_freq=10, log_dir=monitor_dir)
# model.learn(total_timesteps=500_000, callback=callback)

#model.save("ppo_trade")



def ptest():
    env = create_env({
        "window_size": 25
    })
    #model = PPO.load(monitor_dir+"best_model.zip")
    model = PPO.load(monitor_dir + "trial_5_best_model.zip")
    episode_reward = 0
    done = False
    obs, info = env.reset()
    step = 0
    prev_a = 0
    prev_r = 0.0
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        episode_reward += reward
    print("episode_reward = ", episode_reward)
    env.render()

if __name__ == '__main__':
    optimize()
    #ptest()