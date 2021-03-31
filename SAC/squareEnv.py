import sys

import gym
import numpy as np
import gym.spaces
from gym.utils import seeding
import torch

import pybullet as p
import cv2
import math

import random
import copy

import bullet_lidar
import sim  

class squareEnv(gym.Env):
    global_id = 0

    def __init__(self):
        super().__init__()
        self.seed(seed=random.randrange(10000))
        self.sim = None

    def setting(self, _id=-1, mode=p.DIRECT, sec=0.1):
        if _id == -1:
            self.sim = sim.sim_square(squareEnv.global_id, mode, sec)
            squareEnv.global_id += 1
        else:
            self.sim = sim.sim_square(_id, mode, sec)

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        self.lidar = self.createLidar()

        # self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=self.lidar.shape)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(self.lidar.shape[0]+4,))

        self.sec = sec

        self._max_episode_steps = 1000

        self.reset()

    def copy(self, _id=-1):
        new_env = squareEnv()
        
        if _id == -1:
            new_env.sim = self.sim.copy(squareEnv.global_id)
            squareEnv.global_id += 1
        else:
            new_env.sim = self.sim.copy(_id)

        new_env.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)
        new_env.lidar = new_env.createLidar()
        # new_env.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=new_env.lidar.shape)
        new_env.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(new_env.lidar.shape[0]+4,))

        new_env.sec = self.sec

        return new_env

    def reset(self):
        assert self.sim is not None, print("call setting!!") 
        self.sim.reset(sec=self.sec)
        return self.observe()

    def test_reset(self):
        assert self.sim is not None, print("call setting!!") 
        self.sim.test_reset(sec=self.sec)
        return self.observe()

    def createLidar(self):
        # resolusion = 12
        resolusion = 36
        deg_offset = 90.
        rad_offset = deg_offset*(math.pi/180.0)
        startDeg = -180. + deg_offset
        endDeg = 180. + deg_offset

        # maxLen = 20.
        maxLen = 10.
        minLen = 0.
        return bullet_lidar.bullet_lidar(startDeg, endDeg, resolusion, maxLen, minLen)

    def step(self, action):

        done = self.sim.step(action)

        observation = self.sim.observe(self.lidar)
        self.min_obs = np.min(observation) * self.lidar.maxLen

        reward = self.get_reward()

        return observation, reward, done, {}

    def observe(self):
        return self.sim.observe(self.lidar)

    def observe2d(self):
        return self.sim.observe2d(self.lidar)

    def get_reward(self):
        isComtact = self.sim.isContacts()
        isArrive = self.sim.isArrive()

        rewardContact = -1.0 if isComtact else 0.0

        rewardArrive = 1.0 if isArrive else 0.0

        # rewardMove = 0.01 * (self.sim.old_distance - self.sim.distance) / self.sec
        rewardMove = 0.1 * (self.sim.old_distance - self.sim.distance) / self.sec
        
        reward = rewardContact + rewardArrive + rewardMove

        return reward

    def render(self, mode='human', close=False):
        return self.sim.render(self.lidar)

    def close(self):
        self.sim.close()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def sample_random_action(self):
        return self.action_space.sample()

    def getState(self):
        return self.sim.getState()


if __name__ == '__main__':
    
    env = squareEnv()
    env.setting()

    i = 0

    while True:
        i += 1
        
        action = np.array([1.0, 1.0, 1.0])

        _, _, done, _ = env.step(action)

        # cv2.imshow("env", env.render())
        if done or cv2.waitKey(1) >= 0:
            print(i)
            break