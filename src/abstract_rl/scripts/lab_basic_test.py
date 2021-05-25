import gym
import time
import quanser_robots
from quanser_robots import GentlyTerminating

# Cart Pole
CartPole = ['CartpoleStabShort-v0', 
            'CartpoleStabLong-v0',
            'CartpoleStabRR-v0',
            'CartpoleSwingShort-v0',
            'CartpoleSwingLong-v0',
            'CartpoleSwingRR-v0']
# Furuta Pendulum
Qube = ['Qube-v0',
        'QubeRR-v0',
        'QubeRR-v1']
# Magnetig Levitation - NO rendering, only simulation
Levitation = ['Levitation-v0',
              'LevitationRR-v0']
# Basic Pendulum
Pendulum = ['Pendulum-v0']

env_name = CartPole[5]
# env = gym.make(env_name)
env = GentlyTerminating(gym.make(env_name))
for i_episode in range(1):
    print(f'Run {i_episode}')
    obs = env.reset()
    done = False
#     for t in range(100):
    t = 0
    sum_r = 0
    while (not done) and t < 3:
        # env.render()
        # print(obs)
        act = env.action_space.sample()
        obs, rwd, done, info = env.step(act)
        # print(rwd)
        sum_r += rwd
        # time.sleep(0.1)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
        t+=1
    print(sum_r)
