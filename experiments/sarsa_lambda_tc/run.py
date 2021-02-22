import json
from pathlib import Path

from grl.generalized_experiment import GeneralizedExperiment
# from grl.agents import SarsaLambdaTCAgent
from grl.agents import SarsaTCAgent
from grl.envs.mountaincar import MountainCarEnv
from definitions import ROOT_DIR

if __name__ == "__main__":
    # So here we need to run multiple runs over multiple hyperparams.

    # THIS IS JUST A TEST
    # We set num_actions in experiment.py
    agent_hps = {
        'epsilon': 0.01,
        'step_size': 0.125,
        'discount': 0.99,
        'iht_size': 1024,
        'lambda': 0.9,
        'num_tilings': 8,
        'num_tiles': 10
    }

    run_hps = {
        'log_every': 100,
        'max_eps_steps': float('inf'),
        'max_total_steps': 70000
    }

    env_hps_fname = Path(ROOT_DIR, 'experiments', 'tuning_params.json')
    with open(env_hps_fname, 'r') as f:
        env_hpses = json.load(f)

    # TODO: run across all hyperparams

    exp = GeneralizedExperiment(SarsaTCAgent, MountainCarEnv,
               agent_hps=agent_hps, env_hpses=env_hpses, run_hps=run_hps,
               seeds=[2020])

    exp.run()

    print("done")

