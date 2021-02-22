from grl.generalized_experiments import GeneralizedExperiment
from grl.agents import SarsaLambdaTCAgent
from grl.envs.mountaincar import MountainCarEnv

if __name__ == "__main__":
    # So here we need to run multiple runs over multiple hyperparams.

    # THIS IS JUST A TEST
    # We set num_actions in experiment.py
    agent_hps = {
        'epsilon': 0.01,
        'step_size': 1e-3,
        'discount': 0.9,
        'iht_size': 1024,
        'num_tilings': 8,
        'num_tiles': 10
    }

    env_hps = {}

    run_hps = {
        'log_every': 100,
        'max_eps_steps': 500,
        'max_total_steps': 70000
    }



    experiment(SarsaLambdaTCAgent, MountainCarEnv,
               agent_hps=agent_hps, env_hps=env_hps, run_hps=run_hps,
               seeds=[2020])

