"""
Run a set of generalized experiments as per
https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/ADPRL11-shimon.pdf
Essentially runs one set of hyperparameters over a set of "tuning environments"
and returns the hyperparams and agent with the highest total reward averaged
over the tuning environments.
"""
import copy
import numpy as np
from typing import Callable, List

from grl.runner import Runner


class GeneralizedExperiment:
    def __init__(self, agent_class: Callable, env_class: Callable,
                   agent_hps: dict, env_hpses: List[dict], run_hps: dict,
                   seeds: List[int]):
        """
        Run one set of hyperparams for multiple seeds.
        :param agent_class: agent class to instantiate
        :param env_class: environment class to instantiate

        For the below 3 hyperparams, check the respective class for
        an outline of how these hps should look like.
        :param agent_hps: one set of hyperparams to run for the agent.
        :param env_hps: one set of hyperparams to run for the environment.
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


    def run(self):

        for seed in self.seeds:
            agent_hps = copy.deepcopy(self.agent_hps)
            run_hps = copy.deepcopy(self.run_hps)
            agent_hps['seed'] = seed
            run_hps['seed'] = seed

            for original_env_hps in self.env_hpses:
                env_hps = copy.deepcopy(original_env_hps)
                env_hps['seed'] = seed

                env = self.env_class(**env_hps)

                agent_hps['num_actions'] = env.action_space.n
                agent = self.agent_class()
                agent.agent_init(agent_hps)

                runner = Runner(agent, env, run_hps)
                runner.run()

                self.all_avg_ep_rews.append(np.average(runner.all_ep_rewards))

