import gym
import numpy as np

from tqdm import tqdm

from grl.agents import BaseAgent


class Runner:
    def __init__(self, agent: BaseAgent, env: gym.Env, hps: dict):
        """
        Main runner for our RL experiment.
        Initialize with one configuration of agent, environment and hyperparams
        for ONE run.
        then watch'er run!
        :param agent: agent
        :param env: environment
        :param hps: hyperparameters for a SINGLE experiment. Dictionary should contain
        {
            log_every (int): how often to we log?
            max_eps_steps (int): maximum steps to take in an episode
            max_total_steps (int): total number of steps to run for one run.
        }
        """
        self.agent = agent
        self.env = env

        # Should we include this?
        self.max_eps_steps = hps['max_eps_steps']
        self.max_total_steps = hps['max_total_steps']
        # self.max_total_episodes = hps['max_total_episodes']
        self.log_every = hps['log_every']

        self.total_steps = 0
        self.total_episodes = 0
        self.ep_reward = 0
        self.all_ep_rewards = []
        self.pbar = tqdm(total=self.max_total_steps)

        # For stuff you want to log
        # initialize with error first
        self.logs = {
            'error': []
        }

    def run(self):
        # Keep running for max_total_steps
        # while self.total_steps < self.max_total_steps:
        while self.total_steps < self.max_total_steps:
            eps_steps = 0
            self.ep_reward = 0
            obs = self.env.reset()
            done = False

            action = self.agent.agent_start(obs)

            while True:
                obs, rew, done, info = self.env.step(action)

                self.ep_reward += rew
                self.total_steps += 1
                eps_steps += 1

                if not done and eps_steps < self.max_eps_steps and self.total_steps < self.max_total_steps:
                    action, td_error = self.agent.agent_step(rew, obs)
                    self.logs['error'].append(td_error)

                    # Logging per log_every steps
                    if self.total_steps % self.log_every == 0:
                        self.log_error()
                else:
                    self.all_ep_rewards.append(self.ep_reward)
                    break

            td_error = self.agent.agent_end(rew, obs)
            self.logs['error'].append(td_error)
            self.total_episodes += 1

        self.pbar.close()

    def log_error(self):
        self.pbar.update(self.log_every)
        self.pbar.set_description(f'Episode: {self.total_episodes}, '
                                  f'Avg. reward per episode: {np.average(self.all_ep_rewards):.{1}} '
                                  f'TD error: {np.average(self.logs["error"][-100:]):.{3}}')
        # TODO: log errors systematically

