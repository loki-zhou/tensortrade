import tensorflow as tf


# Model Parameters

PPO_PARAMS = {
    '''
    A dictionary for all possible parameters for the PPO Agents
    '''
    'greedy_eval' : 0, # Bool
    'importance_ratio_clipping' : 0.2, # Float ~also clip_ratio
    'lambda_value' : 0.95, # Float ~ also lam
    'discount_factor' : 0.99, # Float
    'entropy_regularization' : 0.0, # Float
    'policy_l2_reg' : 0,  # Float
    'value_function_l2_reg' : 0,  # Float
    'shared_vars_l2_reg' : 0,  # Float
    'value_pred_loss_coef' : 0.5, # Float
    'use_gae' : True, # Bool
    'use_td_lambda_return' : True, #Bool
    'normalize_rewards' : False,  # Bool
    'reward_norm_clipping' : 0,  # Float
    'normalize_observations' : False,  # Bool
    'log_prob_clipping' : 0.2, # Float
    'gradient_clipping' : 0.5, #Float
    'value_clipping' : 0, #Float

    # Use False if I want to use agent.policy instead of agent.collect_policy in training
    'update_normalizers_in_train' : False,
    # Should be set to False when PPOLearner is used.
    'aggregate_losses_across_replicas' : False, # Bool
    'kl_cutoff_factor' : 2.0,
    'kl_cutoff_coef' : 100.0,
    'initial_adaptive_kl_beta' : 1.0,
    'adaptive_kl_target' : 0.01,
    'adaptive_kl_tolerance' : 0.3,
    # Learning parameters
    'epsilon':1e-5,
    'learning_rate' : 5e-06,

    # Modified from PPO Actor and Value Distribution Networks to Actor and Value RNN due to errors with the creation of the sequential PPO Actor network 

    # Distribution RNN Network Parameters
    #Actor RNN Network Parameters
    'arnn_input_fc_layer_params' : (200, 100),
    'arnn_output_fc_layer_params' : (200, 100),
    'arnn_activation_fn' : tf.keras.activations.tanh,
    'arnn_lstm_size' : (20,),
    #'arnn_input_dropout_layer_params' : None,
    #'arnn_preprocessing_layers' : None,
    #'arnn_preprocessing_combiner' : None,
    #'arnn_conv_layer_params' : None,

    # Below parameters causing errors pointing to categorical_projection_network module within actor_distribution_rnn_network module:
    # Commenting them out doesn't help
    #'arnn_discrete_projection_net' : actor_distribution_rnn_network._categorical_projection_net
    #'arnn_continuous_projection_net' : actor_distribution_rnn_network._normal_projection_net
    #'arnn_rnn_contruction_fn' : None
    #'arnn_rnn_contruction_kwargs' : {}

    # Value RNN Network Parameters
    'vrnn_input_fc_layers' : (200,100),
    'vrnn_output_fc_layers' : (200,100),
    'vrnn_input_dropout_layer_params' : None,
    'vrnn_activation_fn' : tf.keras.activations.tanh,
    'vrnn_lstm_size' : (20,),
    'vrnn_preprocessing_layers' : None,
    'vrnn_preprocessing_combiner' : None,
    'vrnn_conv_layer_params' : None ,
}


