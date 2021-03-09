"""
Same as GeneralizedExperiment, but runs each environment in a separate subprocess
"""
import copy
import numpy as np
import multiprocessing as mp
from typing import Callable, List

from grl.runner import Runner


class GeneralizedMPExperiment:
    def __init__(self, agent_class: Callable, env_class: Callable,
                 agent_hps: dict, env_hpses: List[dict], run_hps: dict,
                 seeds: List[int]):
        """
        Run one set of hyperparams for multiple seeds across multiple environments.
        One generalized environment is defined as a set of given hyperparameters.

        :param agent_class: agent class to instantiate
        :param env_class: environment class to instantiate

        For the below 3 hyperparams, check the respective class for
        an outline of how these hps should look like.
        :param agent_hps: one set of hyperparams to run for the agent.
        :param env_hpses: a list of environment hyperparameters to use for each generalized environment.
        :param run_hps: one set of hyperparams for the run itself.

        :param seeds: list of seeds to run each run
        :return: not sure yet lmao
        """
        self.agent_class = agent_class
        self.env_class = env_class
        self.agent_hps = agent_hps
        self.env_hpses = env_hpses
        self.run_hps = run_hps
        self.seeds = seeds

        self.all_avg_ep_rews = []

    def run_one(self, agent_hps, env_hps, run_hps, seed, env_id):
        env_hps['seed'] = seed

        env = self.env_class(**env_hps)

        agent_hps['num_actions'] = env.action_space.n
        agent_hps['num_states'] = env.observation_space.shape[0]
        agent = self.agent_class()
        agent.agent_init(agent_hps)

        runner = Runner(agent, env, run_hps, id=env_id)
        runner.run()

        return np.average(runner.all_ep_rewards)


    def run(self):

        for seed in self.seeds:
            agent_hps = copy.deepcopy(self.agent_hps)
            run_hps = copy.deepcopy(self.run_hps)
            agent_hps['seed'] = seed
            run_hps['seed'] = seed

            param_list = [(agent_hps, env_hps, run_hps, seed, i) for i, env_hps in enumerate(self.env_hpses)]

            with mp.Pool() as p:
                res_list = p.starmap(self.run_one, param_list)

            self.all_avg_ep_rews += res_list

