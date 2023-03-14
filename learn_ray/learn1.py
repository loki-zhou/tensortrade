 # -*- coding: utf-8 -*-

import ray
from ray import tune
import time
ray.shutdown()
ray.init(ignore_reinit_error=True)

config = {
    "env": 'CartPole-v0',
    "num_workers": 10,
    "framework":"torch",
    "num_gpus":1,
    #将要调整的参数以tune.grid_search([])的形式表示，tune即可自动在其中选择最优的参数
    #具体的参数意义可直接在谷歌搜  参数名 ray 即可
    "vf_share_layers": tune.grid_search([True, False]),
    "lr": tune.grid_search([1e-4, 1e-5, 1e-6]),
    }

results = tune.run(
    'PPO',
    stop={
        'timesteps_total': 100000
    },
    #verbose=0,
    config=config)

# =============================================================================
# config = {
#     'env': 'CartPole-v0',
#     'framework':'torch',
#     'num_workers':23,
#     'num_gpus':1,
# }
# stop = {
#     'episode_reward_mean': 200
# }
# st=time.time()
# results = tune.run(
#     'PPO', # Specify the algorithm to train
#     config=config,
#     stop=stop
# )
# =============================================================================
#print('elapsed time=',time.time()-st)

ray.shutdown()