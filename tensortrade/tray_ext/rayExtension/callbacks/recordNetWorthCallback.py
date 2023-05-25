"""Example of using RLlib's debug callbacks.
Here we use callbacks to track the average CartPole pole angle magnitude as a
custom metric.
"""

from typing import Dict

from ray.rllib.algorithms.callbacks import DefaultCallbacks

from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
import pandas as pd
from ray.rllib.algorithms.algorithm import Algorithm
#from tensortrade.oms.wallets.portfolio import Portfolio
# For the moment, keep this inactive an monitor performance


class RecordNetWorthCallback(DefaultCallbacks):

    def on_episode_start(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):

        port = worker.env.action_scheme.portfolio
        port_perform =port.performance

        #performance = pd.DataFrame.from_dict(port_perform).T
        #net_worth = performance['net_worth'].iloc[-1] # Last Output

        net_worth = [nw['net_worth'] for nw in port_perform.values()]

        #net_worth = port.net_worth
        episode.custom_metrics["net_worth"] = net_worth
        # print(f"net_worth: {net_worth}")

    def on_episode_step(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        
        port = worker.env.action_scheme.portfolio
        port_perform =port.performance

        #performance = pd.DataFrame.from_dict(port_perform).T
        #net_worth = performance['net_worth'].iloc[-1] # Last Output

        net_worth = [nw['net_worth'] for nw in port_perform.values()]

        #net_worth = port.net_worth
        episode.custom_metrics["net_worth"] = net_worth
        # print(f"net_worth: {net_worth}")

    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        
        port = worker.env.action_scheme.portfolio
        port_perform =port.performance

        #performance = pd.DataFrame.from_dict(port_perform).T
        #net_worth = performance['net_worth'].iloc[-1] # Last Output

        net_worth = [nw['net_worth'] for nw in port_perform.values()]

        #net_worth = port.net_worth
        episode.custom_metrics["net_worth"] = net_worth
        # print(f"net_worth: {net_worth}")
    '''
    These last four methods have a problematic error preventing me from working with them yet they might me useful for the future.

    TypeError: RecordNetWorthCallback.on_train_result() missing 2 required keyword-only arguments: 'episode' and 'worker'
    TypeError: RecordNetWorthCallback.on_evaluate_start() got an unexpected keyword argument 'algorithm'
    TODO: (GBB) Find out why the error is persistent

    def on_evaluate_start(
        self,
        worker: RolloutWorker,
        episode: Episode
    ) -> None:
        
        port = worker.env.action_scheme.portfolio
        port_perform =port.performance

        #performance = pd.DataFrame.from_dict(port_perform).T
        #net_worth = performance['net_worth'].iloc[-1] # Last Output

        net_worth = [nw['net_worth'] for nw in port_perform.values()]

        #net_worth = port.net_worth
        episode.custom_metrics["net_worth"] = net_worth
        # print(f"net_worth: {net_worth}")

    def on_evaluate_end(
        self,
        worker: RolloutWorker,
        episode: Episode,
    ) -> None:
        
        port = worker.env.action_scheme.portfolio
        port_perform =port.performance

        #performance = pd.DataFrame.from_dict(port_perform).T
        #net_worth = performance['net_worth'].iloc[-1] # Last Output

        net_worth = [nw['net_worth'] for nw in port_perform.values()]

        #net_worth = port.net_worth
        episode.custom_metrics["net_worth"] = net_worth
        # print(f"net_worth: {net_worth}")
    
    def on_sample_end(
        self,
        *,
        worker: RolloutWorker,
        episode: Episode,
        **kwargs
    ):

        port = worker.env.action_scheme.portfolio
        port_perform =port.performance

        #performance = pd.DataFrame.from_dict(port_perform).T
        #net_worth = performance['net_worth'].iloc[-1] # Last Output

        net_worth = [nw['net_worth'] for nw in port_perform.values()]

        #net_worth = port.net_worth
        episode.custom_metrics["net_worth"] = net_worth
        # print(f"net_worth: {net_worth}")
        

    def on_train_result(
        self,
        *,
        algorithm: 'Algorithm',
        result: dict,
        # The below params not in official method
        # However, monitor whether it works since the method can be overriden to implement custom logic
        episode: Episode, 
        worker: RolloutWorker,
        **kwargs
    ):
    
        port = worker.env.action_scheme.portfolio
        port_perform =port.performance

        #performance = pd.DataFrame.from_dict(port_perform).T
        #net_worth = performance['net_worth'].iloc[-1] # Last Output

        net_worth = [nw['net_worth'] for nw in port_perform.values()]

        #net_worth = port.net_worth
        episode.custom_metrics["net_worth"] = net_worth
        # print(f"net_worth: {net_worth}")
    '''
'''
Notes:
_____________
Trial 1:
----
Include the record networth on all the methods in this file using original networth metric

Trial 2:
----
Trial 2.1:
    net_worth = [nw['net_worth'] for nw in portfolio.performance.values()]

Trial 2.2
    performance = pd.DataFrame.from_dict(portfolio.performance).T
    net_worth = performance['net_worth'].iloc[-1] # Last Output

LOGIC
--------
I need to understand how the model is performing during tuning, training and evaluation
'''