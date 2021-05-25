import sys
sys.path.append('.')

from abstract_rl.src.run.run_mpo_non_parametric import run_mpo_non_parametric


run_config = {

    # simulations ettings
    'num_epochs': 1000,
    'max_steps': 20000,
    'seed': 22,
    'reload': False,
    'use_cuda': True,
    ''

    # environment settings
    'env_name': 'CartpoleSwingShort-v0',
    # 'env_name': 'CartpoleStabLong-v0',
    # 'env_name': 'CartpoleStabRR-v0',
    # 'env_name': 'CartpoleSwingShort-v0',
    # 'env_name': 'CartpoleSwingLong-v0',
    # 'env_name': 'CartpoleSwingRR-v0',
    'env_angle_to_sin_cos': 0,
    'env_traj_evaluations': 1,
    'env_discount': 0.99,
    'env_render': False,
    'env_num_threads': 1,
    'env_normalize': True,
    'env_sample': 3000,

    # display and environment
    'alg': 'mpo',

    # create off policy memory
    'tc_size': 2000000,
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
    'val_opt_steps': 5,
    'val_batch_size': 64,
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
    'policy_opt_steps': 100,
    'policy_sigma': 2.0,
    'policy_cov_type': 'single',
    'policy_batch_size': 64,
    'filter_states': False,
    'filter_rewards': False,

    'mpo_inner_lr': 0.0003,
    'mpo_outer_lr': 0.0003,
    'mpo_l2_reg': 0.0001,
    'mpo_mode': 'non-parametric',
    'mpo_trust_regions': [0.001, 0.0005],
    'mpo_coordinate_ascent_objective': False,
    'mpo_coordinate_ascent_constraint': False
}

run_mpo_non_parametric(run_config)
