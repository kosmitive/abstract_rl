import gym
import torch
import torch.optim as opt

from src.sac.gaussian_policy import GaussianPolicy
from src.sac.q_network import QNetwork
from src.sac.replay_memory import ReplayMemory
from src.sac.rl_util import tt, tn
from src.sac.v_network import VNetwork

torch.autograd.set_detect_anomaly(True)

env = gym.make('Pendulum-v0')
hidden_dim = 128

policy = GaussianPolicy(env, hidden_dim)
v_net = VNetwork(env, hidden_dim)
target_v_net = v_net.clone()
q_net = QNetwork(env, hidden_dim)
mem = ReplayMemory(env, 2000)

cur_st = env.reset()

num_iterations = 1000000
batch_size = 64
discount = 0.99
training_pause = 100
training_updates = 1
alpha = 0.2

# optimizers
opt_v = opt.Adam(v_net.parameters(), lr=0.005)
opt_q1 = opt.Adam(q_net.params_q1(), lr=0.005)
opt_q2 = opt.Adam(q_net.params_q2(), lr=0.005)
opt_p = opt.Adam(policy.parameters(), lr=0.005)

for t in range(num_iterations):

    # make a step
    act, _, _ = policy.sample(cur_st)

    # obtain next samples
    nxt_st, rew, done, _ = env.step(tn(act))
    mem.insert(cur_st, tn(act), rew, done, nxt_st)

    # close
    np_curs = env.reset() if done else nxt_st

    env.render()

    # train after some steps were executed
    if t % training_pause == training_pause - 1:

        # perform the learning
        for t in range(training_updates):

            # sample a batch
            states, actions, rews, dones, nxt_states = mem.sample(batch_size)
            smp_actions, smp_lls, smp_act_means = policy.sample(states)

            # compute targets
            q_target = rews + discount * (1 - dones) * target_v_net(nxt_states)
            q_val = torch.stack(list(q_net(states, smp_actions)), dim=-1)
            v_target = torch.min(q_val, dim=-1)[0] - alpha * smp_lls
            q_target = q_target.detach()
            v_target = v_target.detach()

            # update q1
            opt_q1.zero_grad()
            q_val_1, _ = q_net(states, actions)
            loss_q1 = torch.mean((q_val_1 - q_target).pow(2))
            loss_q1.backward()
            opt_q1.step()

            # and q2
            opt_q2.zero_grad()
            _, q_val_2 = q_net(states, actions)
            loss_q2 = torch.mean((q_val_2 - q_target).pow(2))
            loss_q2.backward()
            opt_q2.step()

            # update v
            opt_v.zero_grad()
            v_val = v_net(states)
            loss_v = torch.mean((v_val - v_target).pow(2))
            loss_v.backward()
            opt_v.step()

            # update policy
            opt_p.zero_grad()
            q_val_1, q_val_2 = q_net(states, smp_actions)
            loss_p = torch.mean((q_val_1 - alpha * smp_lls))
            loss_p.backward()
            opt_p.step()

            target_v_net.exp_avg(0.995, v_net.psi())

env.close()