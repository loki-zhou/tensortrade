from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print


# algo = (
#     PPOConfig()
#     .rollouts(num_rollout_workers=1)
#     .resources(num_gpus=0)
#     .environment(env="CartPole-v1")
#     .build()
# )
#
# for i in range(10):
#     result = algo.train()
#     print(pretty_print(result))
#
#     if i % 5 == 0:
#         checkpoint_dir = algo.save()
#         print(f"Checkpoint saved in directory {checkpoint_dir}")
#


# Note: `gymnasium` (not `gym`) will be **the** API supported by RLlib from Ray 2.3 on.

import gymnasium as gym


from ray.rllib.algorithms.algorithm import Algorithm

from ray.rllib.algorithms.ppo import PPOConfig

env_name = "CartPole-v1"
env = gym.make(env_name)
#algo = PPOConfig().environment(env_name).build()
checkpoint_path = r"C:\Users\zhouyi/ray_results\PPO_CartPole-v1_2023-03-17_17-56-59z3lpriyp\checkpoint_000006"
algo = Algorithm.from_checkpoint(checkpoint_path)
episode_reward = 0
terminated = truncated = False


obs, info = env.reset()

while not terminated and not truncated:
    action = algo.compute_single_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    episode_reward += reward
print("episode_reward = ", episode_reward)