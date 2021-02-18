from typing import Callable, List

from grl.runner import Runner

def experiment(agent_class: Callable, env_class: Callable,
               agent_hps: dict, env_hps: dict, run_hps: dict,
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
    for seed in seeds:
        agent_hps['seed'] = seed
        env_hps['seed'] = seed
        run_hps['seed'] = seed

        env = env_class(**env_hps)

        agent_hps['num_actions'] = env.action_space.n
        agent = agent_class()
        agent.agent_init(agent_hps)

        runner = Runner(agent, env, run_hps)
        runner.run()

        # TODO: evaluation and data collection.

