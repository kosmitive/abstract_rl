import sys
sys.path.append('.')


from abstract_rl.src.run.run_trpo import run_trpo

run_config = {

    # simulations settings
    'num_epochs': 1000,
    'max_steps': 10000,
    'reload': False,
    'seed': 17,

    'env_name': 'Qube-v0',
    # 'env_name': 'QubeRR-v0',
    'env_discount': 0.995,
    'env_normalize': True,
    'env_center_states': False,
    'env_num_threads': 1,
    'env_render': False,

    # fully connected neural q function
    'val_struct': [64, 64],
    'val_lr': 0.00001,
    'val_steps_per_epoch': 50,
    'val_layer_norm': False,
    'val_act_fn': ['tanh', 'tanh', 'none'],
    'val_init': 'xavier_gaussian',
    'val_batch_size': 64,
    'val_sync_steps': 1,
    'val_l2_reg': 0.001,

    # advantage settings
    'gae_lambda': 0.97,

    # policy network structure
    'policy_structure': [64, 64],
    'policy_act_fn': ['tanh', 'tanh', ['none', 'softplus']],
    'policy_layer_norm': False,
    'policy_opt_steps': 5,
    'policy_init': 'gaussian',
    'policy_sync_steps': 1,
    'policy_batch_size': 16392,
    'policy_sigma': 1.0,
    'policy_cov_type': 'single',
    'filter_states': False,
    'filter_rewards': False,

    # settings for trpo
    'n_bootstrap_steps': 50,
    'trpo_cg_max_k': 10,
    'trpo_cg_res_tol': 1e-10,
    'trpo_dvp_damping': 0.1,
    'trpo_tr_eps': 0.001

}
run_trpo(run_config)
