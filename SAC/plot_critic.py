import gym
import pybullet_envs
from SAC import SAC
from trainer import Trainer

from mazeEnv import mazeEnv 
from crossEnv import crossEnv 
from squareEnv import squareEnv 
from maze3Env import maze3Env 

import torch

import matplotlib.pyplot as plt
import numpy as np



# ENV_ID = 'InvertedPendulumBulletEnv-v0'
SEED = 0
REWARD_SCALE = 1.0
# NUM_STEPS = 5 * 10 ** 4
# NUM_STEPS = 10 * 10 ** 4
NUM_STEPS = 2 * 10 ** 5
EVAL_INTERVAL = 10 ** 3

# env = gym.make(ENV_ID)
# env_test = gym.make(ENV_ID)

# env = mazeEnv()
# env = crossEnv()
# env = squareEnv()
env = maze3Env()
env.setting()

# env_test = mazeEnv()
# env_test = crossEnv()
# env_test = squareEnv()
env_test = maze3Env()
env_test.setting()

algo = SAC(
    state_shape=env.observation_space.shape,
    action_shape=env.action_space.shape,
    seed=SEED,
    reward_scale=REWARD_SCALE,
)

algo.load()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# xs = np.arange(0.0, 9.0+1e-6, 1.0)
# ys = np.arange(0.0, 9.0+1e-6, 1.0)

xs = np.arange(0.0-3.0, 9.0+1e-6+3.0, 0.5)
ys = np.arange(0.0-3.0, 9.0+1e-6+3.0, 0.5)

print(xs, ys)

data = np.zeros((len(xs), len(ys)))

j = 0

sx = 8.0
sy = 7.5

def draw_heatmap(data, row_labels, column_labels):
    # 描画する
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(data, cmap=plt.cm.Blues)

    ax.set_xticks(np.arange(data.shape[0]) + 0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[1]) + 0.5, minor=False)

    # ax.invert_yaxis()
    # ax.xaxis.tick_top()

    ax.set_xticklabels(row_labels, minor=False)
    ax.set_yticklabels(column_labels, minor=False)
    plt.savefig('image{}_{}_ex.png'.format(sx, sy))
    plt.show()

    return heatmap

for y in ys:
    
    i = 0
    
    for x in xs:

        env_test.sim.test_reset2(sec=0.01, sx=sx, sy=sy, gx=float(x), gy=float(y))

        state = env_test.observe()
        action = algo.exploit(state)

        states = torch.tensor(state).to(device)
        actions = torch.tensor(action).to(device)
        q1, q2 = algo.critic(states, actions)
        q = torch.min(q1, q2)

        # print(q.detach().cpu().numpy())
        data[i][j] = q.detach().cpu().numpy()[0]
        # print(i, j, x, y)

        i += 1
    j += 1

print(data)
draw_heatmap(data, xs, ys)
