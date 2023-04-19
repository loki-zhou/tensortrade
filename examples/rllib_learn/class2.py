import gymnasium
import ray
import numpy as np
import pandas as pd
import ray.rllib.algorithms.ppo as ppo

from ray import tune, air
from ray.tune.registry import register_env
#Checkpoint(local_path=C:\Users\zhouyi\ray_results\PPO\PPO_CartPole-v1_12831_00000_0_lr=0.0100_2023-04-17_21-09-23\checkpoint_000007)
checkpoint_path = r"C:\Users\zhouyi\ray_results\PPO\PPO_CartPole-v1_12831_00000_0_lr=0.0100_2023-04-17_21-09-23\checkpoint_000007"
ray.init()
from ray.rllib.algorithms.algorithm import Algorithm
agent = Algorithm.from_checkpoint(checkpoint_path)
print(f"Agent loaded from saved model at {checkpoint_path}")

env = gymnasium.make("CartPole-v1")
# Run until episode ends
episode_reward = 0
done = False
obs, info = env.reset()
step = 0
prev_a = 0
prev_r = 0.0
for i in range(10000):
    action = agent.compute_single_action(obs)
    obs, reward, terminated, truncated, info  = env.step(action)
    episode_reward += reward
    env.render()
    if terminated or truncated:
        print(f"Cart pole ended after {i} steps.")
        break
print("episode_reward = ", episode_reward)

ray.shutdown()