
import os
import shutil
import time

import ray
from ray import tune
from ray.tune import ExperimentAnalysis
from ray.tune.stopper import (
  CombinedStopper,
  MaximumIterationStopper,
  TrialPlateauStopper,
)

# Tune CallBack
from tensortrade.tray_ext.rayExtension.callbacks.printCallback import PrintCallback
from tensortrade.tray_ext.rayExtension.callbacks.renderCallback import RenderCallback

# Tune Stopper
from tensortrade.tray_ext.rayExtension.stoppers.netWorthstopper import NetWorthstopper


class Strategy():
  """
  A base strategy class brinigs together the algorithm config and an environment.
    - Training,
    - Evaluation
  """
  def __init__(self) -> None:
    self.max_epoch = None
    self.evaluation_frequency = None
    self.net_worth_threshold = None
    self.patience = 0
    #self.num_samples = 10
    
    self.config = None
    self.env_train_config = None
    self.agent = None
    raise NotImplementedError

  def train(self) -> ExperimentAnalysis:
    '''
      Train the RL agent for this strategy
    '''
    # Register the environment
    tune.register_env("TradingEnv", self.create_env)

    # dashboard
    # (Re)Start the ray runtime.
    if ray.is_initialized():
      ray.shutdown()

    ray.init(
      #local_mode=True, # Deprecated and will be removed
      include_dashboard=True,
      ignore_reinit_error=True,
      num_gpus=1,
      num_cpus=4, # Intel Core i5-8350U 4 cores 6MB Cache 1.7 - 1.9 Ghz
      #dashboard_host="120.0.0.7",
      dashboard_port=8081,
      )
    
    start = time.time()

    # Setup stopping conditions
    stopper = CombinedStopper(
        MaximumIterationStopper(max_iter=self.max_epoch),
        NetWorthstopper(net_worth_mean=self.net_worth_threshold, patience=self.patience),
        TrialPlateauStopper(metric="net_worth_max")
    )

    # train an agent
    analysis = tune.run(
      run_or_experiment= self.algorithm_name,
      name=self.log_name,
      metric="episode_reward_mean",
      mode="max",
      stop=stopper,
      #time_budget_s
      config=self.config,
      #resources_per_trial={"cpu": 1, "gpu": 0},
      #num_samples=self.num_samples,
      #local_dir=self.local_dir,
      #search_alg=,
      #scheduler=,
      #keep_checkpoints_num=,
      checkpoint_freq=1,
      checkpoint_at_end=True,
      verbose=1,
      #progress_reporter=,
      #log_to_file=.,
      #trial_name_creator=,
      #trial_dirname_creator=,
      #chdir_to_trial_dir=,
      #sync_config=,
      #export_formats=,
      #max_failures=,
      #fail_fast=,
      #restore=,
      #server_port=,
      #resume=,
      #reuse_actors=,
      #raise_on_failed_trial=,
      callbacks=[
        PrintCallback(),
        RenderCallback(
          self.evaluation_frequency,
          self.log_name,
          self.log_dir)]
      #max_concurrent_trials=,
      #trial_executor=,
      #_experiment_checkpoint_dir: str | None = None,
      #_remote: bool | None = None,
      #_remote_string_queue: Queue | None = None
    )
    print(f"Best Trail log directory: {analysis.best_logdir}")
    ray.shutdown()

    taken = time.time() - start
    print(f"Time taken: {taken:.2f} seconds.")

    self.best_logdir = analysis.best_trial.checkpoint#.value
    return analysis

  def evaluate(self, best_logdir = None):
    '''
      Evaluate the RL agent for this strategy
    '''
    # Register the environment
    tune.register_env("TradingEnv", self.create_env)

    if not best_logdir:
      best_logdir = self.best_logdir

    # Restore agent
    self.agent(
      env="TradingEnv",
      config=self.config
    )
    self.agent.restore(checkpoint_path= best_logdir)
    # evaluate an episode
    # agent.evaluate()

    # Instantiate the environment
    env = self.create_env(self.env_train_config)
    
    # Run until episode ends
    episode_reward = 0
    done = False
    terminate = False
    truncate = False
    obs = env.reset()

    while not done:
        action = self.agent.compute_single_action(obs) # Signle Obs & Action
        #action = self.agent.compute_actions(obs) # Multiple Obs $ Actions
        
        #obs, reward, terminate, truncate, info= env.step(action) # MOD
        obs, reward, done, info= env.step(action) # Original
        episode_reward += reward

    env.render()

  def getConfig(self):
    return self.config

  def clearLogs(self):
    path = f'{self.log_dir}{self.log_name}'
    if os.path.exists(path):
      shutil.rmtree(path, ignore_errors=False)

'''
Notes
____________
COMPUTE_SONGLE_ACTION VS COMPUTE_ACTIONS
To compute actions for given observations use compute_single_action.
In case you should need to compute many actions at once, not just a single one, you can use the compute_actions method instead, which takes dictionaries of observations as input and produces dictionaries of actions with the same dictionary keys as output.  
I would use compute actions when training the agent on multiple tickers
'''