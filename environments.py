"""
Modified version of classic cart-pole system implemented by Rich Sutton et al.
"""
# flake8: noqa
import math
from typing import Callable, List, Tuple

import gym
import numpy as np
from gym import logger, spaces
from gym.utils import seeding
import matplotlib.pyplot as plt
import os
import matplotlib.animation as animation
import matplotlib.image as mpimg


class CartPoleRegulatorEnv(gym.Env):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum starts upright, and the goal is to prevent it from falling over by increasing and reducing the cart's velocity.
    Source:
        This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson
    Observation:
        Type: Box(4)
        Num Observation                 Min         Max
        0   Cart Position             -4.8            4.8
        1   Cart Velocity             -Inf            Inf
        2   Pole Angle                 -24 deg        24 deg
        3   Pole Velocity At Tip      -Inf            Inf
    Actions:
        Type: Discrete(2)
        Num Action
        0   Push cart to the left
        1   Push cart to the right
        Note: The amount the velocity that is reduced or increased is not fixed; it depends on the angle the pole is pointing. This is because the center of gravity of the pole increases the amount of energy needed to move the cart underneath it
    Reward:
        Reward is 1 for every step taken, including the termination step
    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]
    Episode Termination:
        Pole Angle is more than 12 degrees
        Cart Position is more than 2.4 (center of the cart reaches the edge of the display)
        Episode length is greater than 200
        Solved Requirements
        Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.
    """

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(
            self,
            mode="train",
            force_left=0.0,
            group=0,
            fg_only=False,
            x_success_range=2.4,
            theta_success_range=12 * 2 * math.pi / 360
    ):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = "euler"
        self.unique_actions = np.array([0, 1])
        self.group = group
        self.state_dim = 4
        self.fg_only = fg_only
        self.force_left = force_left
        self.max_steps = 1000

        self.x_success_range = x_success_range
        self.theta_success_range = theta_success_range
        self.x_threshold = 2.4
        self.theta_threshold_radians = math.pi / 2

        self.c_trans = 0.01

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ]
        )

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _compute_next_state(self, state, action):
        x, x_dot, theta, theta_dot = state
        force = self.force_mag if action == 1 else -self.force_mag

        force -= self.force_left

        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (
                       force + self.polemass_length * theta_dot * theta_dot * sintheta
               ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
                self.length
                * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        return x, x_dot, theta, theta_dot

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (
            action,
            type(action),
        )
        self.state = self._compute_next_state(self.state, action)
        x, _, theta, _ = self.state

        self.episode_step += 1

        # Forbidden States (S-)
        if (
                x < -self.x_threshold
                or x > self.x_threshold
                or theta < -self.theta_threshold_radians
                or theta > self.theta_threshold_radians
        ):
            done = True
            cost = 1
        # Goal States (S+)
        elif (
                -self.x_success_range < x < self.x_success_range
                and -self.theta_success_range < theta < self.theta_success_range
        ):
            done = False
            cost = 0
        else:
            done = False
            cost = self.c_trans

        # Check for time limit
        info = {"time_limit": self.episode_step >= self.max_steps}

        return np.array(self.state), cost, done, info

    def reset(self):
        if self.mode == "train" or self.mode == 'eval':
            self.state = self.np_random.uniform(
                low=[-2.3, 0, -0.3, 0], high=[2.3, 0, 0.3, 0], size=(4,)
            )
        else:
            self.state = self.np_random.uniform(low=[-self.x_success_range, 0, -self.theta_success_range, 0],
                                                high=[self.x_success_range, 0, self.theta_success_range, 0], size=(4,))

        self.episode_step = 0

        return np.array(self.state)

    def generate_rollout(
            self, agent=None, render: bool = False, group: int = 0
    ) -> List[Tuple[np.array, int, int, np.array, bool, int]]:
        """
        Generate rollout using given action selection function.
        If a network is not given, generate random rollout instead.
        Parameters
        ----------
        get_best_action : Callable
            Greedy policy.
        render: bool
            If true, render environment.
        Returns
        -------
        rollout : List of Tuple
            Generated rollout.
        episode_cost : float
            Cumulative cost throughout the episode.
        """
        rollout = []
        episode_cost = 0
        obs = self.reset()
        done = False
        info = {"time_limit": False}
        while not done and not info["time_limit"]:
            if agent is not None:
                action = agent.get_best_action(obs, self.unique_actions, group)
            else:
                action = self.action_space.sample()

            next_obs, cost, done, info = self.step(action)
            rollout.append((obs, action, cost, next_obs, done, group))
            episode_cost += cost
            obs = next_obs

            if render:
                self.render()

        return rollout, episode_cost, self.episode_step