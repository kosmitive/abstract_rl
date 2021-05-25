import sys
sys.path.append('.')

from abstract_rl.src.run.run_trpo import run_trpo

run_config = {

    # simulations settings
    'num_epochs': 1000,
    'max_steps': 10000,
    'reload': False,
    'seed': 3,

    'env_name': 'CartpoleStabShort-v0',
    # 'env_name': 'CartpoleStabLong-v0',
    # 'env_name': 'CartpoleStabRR-v0',
    # 'env_name': 'CartpoleSwingShort-v0',
    # 'env_name': 'CartpoleSwingLong-v0',
    # 'env_name': 'CartpoleSwingRR-v0',
    'env_discount': 0.95,
    'env_num_threads': 1,
    'env_render': False,
    'env_traj_evaluations': 5,
    'env_normalize': True,

    # fully connected neural q function
    'val_struct': [64, 64],
    'val_tr_eps': 1e-3,
    'val_cg_res_tol': 1e-8,
    'val_cg_max_k': 10,
    'val_dvp_damping': 0.001,
    'val_steps_per_epoch': 1,
    'val_layer_norm': False,
    'val_act_fn': ['tanh', 'tanh', 'none'],
    'val_init': 'xavier_uniform',
    'val_batch_size': 512,
    'val_sync_steps': 1,
    'val_l2_reg': 0.001,

    # advantage settings
    'gae_lambda': 0.99,

    # policy network structure
    'policy_structure': [64, 64],
    'policy_act_fn': ['tanh', 'tanh', ['none', 'softplus']],
    'policy_layer_norm': False,
    'policy_steps_per_epoch': 1,
    'policy_init': 'xavier_uniform',
    'policy_sync_steps': 1,
    'policy_batch_size': 512,
    'policy_cov_type': 'single',
    'policy_sigma': 0.01,
    'filter_states': False,
    'filter_rewards': False,

    # settings for trpo
    'n_bootstrap_steps': 50,
    'trpo_cg_max_k': 10,
    'trpo_cg_res_tol': 0.00001,
    'trpo_dvp_damping': 0.01,
    'trpo_tr_eps': 1e-5

}
run_trpo(run_config)
