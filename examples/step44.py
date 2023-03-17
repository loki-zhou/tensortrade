import gymnasium as gym
from ray.rllib.algorithms.ppo import PPOConfig

env_name = "CartPole-v1"
env = gym.make(env_name)
algo = PPOConfig().environment(env_name).build()
episode_reward = 0
terminated = truncated = False

obs, info = env.reset()

while not terminated and not truncated:
    action = algo.compute_single_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    episode_reward += reward
