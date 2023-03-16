from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print


# algo = (
#     PPOConfig()
#     .rollouts(num_rollout_workers=1)
#     .resources(num_gpus=0)
#     .environment(env="CartPole-v1")
#     .training()
#     .build()
# )
# print(algo)

# for i in range(10):
#     result = algo.train()
#     print(pretty_print(result))
#
#     if i % 5 == 0:
#         checkpoint_dir = algo.save()
#         print(f"Checkpoint saved in directory {checkpoint_dir}")

import ray
from ray import air, tune

ray.init()

config = PPOConfig().environment(env="CartPole-v1").framework("torch").training(lr=tune.grid_search([0.01, 0.001, 0.0001]))

tuner = tune.Tuner(
    "PPO",
    run_config=air.RunConfig(
        stop={"episode_reward_mean": 150},
    ),
    param_space=config,
)

tuner.fit()