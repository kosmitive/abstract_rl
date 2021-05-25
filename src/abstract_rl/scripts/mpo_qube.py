import sys
sys.path.append(".")

from abstract_rl.src.run.run_mpo_non_parametric import run_mpo_non_parametric


run_config = {

    # simulations ettings
    'num_epochs': 10000,
    'max_steps': 300,
    'seed': 127,
    'reload': False,

    # environment settings
    'env_name': 'Qube-v0',
    'env_angle_to_sin_cos': 0,
    'env_traj_evaluations': 3,
    'env_discount': 0.99,
    'env_render': True,
    'env_num_threads': 6,
    'env_normalize': True,
    'env_sample': 1024,

    # display and environment
    'alg': 'mpo',

    # create off policy memory
    'tc_size': 300000,
    'mem_bootstrap_steps': 5,

    # retrace parameters
    'ret_lambda': 1.0,
    'ret_add_acts': 20,
    'res_add_acts': 20,

    #
    'rnk_eps': 0.001,
    'rnk_lr': 0.0003,
    'rnk_init_eta': 1.0,
    'rnk_max_steps': 20,
    'gae_lambda': 1.0,

    # general feed forward network
    'val_struct': [64, 64],
    'val_lr': 0.00001,
    'val_opt_steps': 50,
    'val_batch_size': 1024,
    'val_layer_norm': True,
    'val_act_fn': ['tanh', 'tanh', 'none'],
    'val_init': 'gaussian',
    'val_sync_steps': 1,
    'val_ret_lambda': 1.0,
    'val_l2_reg': 0.0001,

    # policy network structure
    'policy_structure': [64, 64],
    'policy_act_fn': ['tanh', 'tanh', ['none', 'softplus']],
    'policy_layer_norm': True,
    'policy_init': 'gaussian',
    'policy_sync_steps': 1,
    'policy_opt_steps': 50,
    'policy_sigma': 2.0,
    'policy_cov_type': 'single',
    'policy_batch_size': 1024,
    'filter_states': False,
    'filter_rewards': False,

    'mpo_inner_lr': 0.0003,
    'mpo_outer_lr': 0.0003,
    'mpo_mode': 'non-parametric',
    'mpo_trust_regions': [0.001],
    'mpo_coordinate_ascent_objective': False,
    'mpo_coordinate_ascent_constraint': False
}

run_mpo_non_parametric(run_config)
