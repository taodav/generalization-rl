"""
Generalized Mountain Car. Based on specifications and code adapted from 
https://www.cs.utexas.edu/users/pstone/Papers/bib2html-links/ADPRL11-shimon.pdf
"""
import math
import numpy as np

from gym.envs.classic_control import MountainCarEnv


class GeneralizedMountainCarEnv(MountainCarEnv):
    def __init__(self, goal_velocity=0, accel_bias_mean=1., accel_factor=0.001):
        super(GeneralizedMountainCarEnv, self).__init__(goal_velocity)
        self._accel_bias_mean = accel_bias_mean
        self._accel_factor = accel_factor

    def step(self, action: int):
        adjusted_gaussian = np.random.normal(scale=0.05)
        accel_noise = adjusted_gaussian + self._accel_bias_mean
        varied_accel = self._accel_factor * accel_noise

        self.force = varied_accel
        return super(GeneralizedMountainCarEnv, self).step(action)
        # assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        # 
        # position, velocity = self.state
        # velocity += (action - 1) * self.force + math.cos(3 * position) * (-self.gravity)
        # velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        # position += velocity
        # position = np.clip(position, self.min_position, self.max_position)
        # if (position == self.min_position and velocity < 0)::w

        #     velocity = 0
        # 
        # done = bool(
        #     position >= self.goal_position and velocity >= self.goal_velocity
        # )
        # reward = -1.0
        # 
        # self.state = (position, velocity)
        # return np.array(self.state), reward, done, {}


    def setAccelFactor(self, d: float):
        self._accel_factor = d

    def setAccelBiasMean(self, d: float):
        self._accel_bias_mean = d
