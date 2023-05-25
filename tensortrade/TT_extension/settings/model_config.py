'''
A config dictionary for setting up the model for, training, evaluation, testing and deployment
'''
root_dir = './home/gich2023/Future~WSL/Lanuvo/lengine/Projects/TrainingData'
random_seed = None
# Params for collect
# num_environment_steps=25000000,
collect_episodes_per_iteration = 2
num_parallel_environments = 2
replay_buffer_capacity = 100000  # Per-environment
# Params for eval
num_eval_episodes = 5
eval_interval = 5
# Params for summaries and logging
train_checkpoint_interval = 5
policy_checkpoint_interval = 5
log_interval = 5,
summary_interval = 5
summaries_flush_secs = 1
use_tf_functions = True
num_iterations = 10
use_parallel_envs = True
