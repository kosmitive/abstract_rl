import sys
sys.path.append(".")

from abstract_rl.src.run.run_hpsearch import run_hpsearch
from abstract_rl.src.run.run_trpo import run_trpo


# config used for each run
basic_run_config = {

    # simulations settings
    'num_epochs': 3,
    'max_steps': 10000,
    'reload': False,
    'seed': 97,

    'env_name': 'Qube-v0',
    # 'env_name': 'QubeRR-v0',
    'env_discount': 0.999,
    'env_normalize': True,
    'env_center_states': False,
    'env_num_threads': 1,
    'env_render': False,
    'env_traj_evaluations': 10,

    # fully connected neural q function
    'val_struct': [32, 32],
    'val_tr_eps': 0.001,
    'val_cg_res_tol': 1e-8,
    'val_cg_max_k': 10,
    'val_dvp_damping': 0.001,
    'val_steps_per_epoch': 5,
    'val_layer_norm': True,
    'val_act_fn': ['tanh', 'tanh', 'none'],
    'val_init': 'xavier_gaussian',
    'val_batch_size': 128,
    'val_sync_steps': 1,

    # advantage settings
    'gae_lambda': 0.8,

    # policy network structure
    'policy_structure': [32, 32],
    'policy_act_fn': ['tanh', 'tanh', ['none', 'softplus']],
    'policy_layer_norm': True,
    'policy_opt_steps': 5,
    'policy_init': 'xavier_gaussian',
    'policy_sync_steps': 1,
    'policy_batch_size': 1024,
    'policy_sigma': 3.0,
    'policy_cov_type': 'state',
    'policy_type': 'gaussian',

    # settings for trpo
    'n_bootstrap_steps': 50,
    'trpo_cg_max_k': 10,
    'trpo_cg_res_tol': 0.00001,
    'trpo_dvp_damping': 1e-4,
    'trpo_tr_eps': 0.001

}


# defines hyperparameters to vary / test
hp_search_config = {
    'gae_lambda': [0.4, 0.8, 0.95],
    'val_dvp_damping': [1e-2, 1e-3, 1e-4],
    'trpo_tr_eps': [1e-3, 1e-4, 1e-5, 1e-6, 1e-7],
    'policy_layer_norm': [True, False],
    'policy_batch_size': [256, 512, 1024, 2048]
}


run_hpsearch(run_trpo, basic_run_config, hp_search_config)
