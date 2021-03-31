import gym
import pybullet_envs
from PPO import PPO
from trainer import Trainer

from mazeEnv import mazeEnv 
from crossEnv import crossEnv 
from squareEnv import squareEnv 
from maze3Env import maze3Env 

# ENV_ID = 'InvertedPendulumBulletEnv-v0'
SEED = 0
NUM_STEPS = 5 * 10 ** 4
# NUM_STEPS = 10 * 10 ** 4
# NUM_STEPS = 2 * 10 ** 5
# NUM_STEPS = 25 * 10 ** 4
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

algo = PPO(
    state_shape=env.observation_space.shape,
    action_shape=env.action_space.shape,
    seed=SEED
)

trainer = Trainer(
    env=env,
    env_test=env_test,
    algo=algo,
    seed=SEED,
    num_steps=NUM_STEPS,
    eval_interval=EVAL_INTERVAL,
)

trainer.train()

trainer.plot()

algo.save()

# trainer.visualize()
