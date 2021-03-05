import json
import time
import numpy as np
from pathlib import Path
from itertools import product
from pprint import PrettyPrinter

from grl.generalized_experiment import GeneralizedExperiment
from grl.agents import SarsaTCAgent
from grl.envs.mountaincar import MountainCarEnv
from definitions import ROOT_DIR

if __name__ == "__main__":
    pp = PrettyPrinter(indent=4)
    # So here we need to run multiple runs over multiple hyperparams.

    # step_sizes = [1.0, 0.75, 0.5, 0.25, 0.125, 0.06125]
    #
    # lambdas = [0.99, 0.95, 0.9, 0.75, 0.5, 0.25, 0.125, 0.06125, 0]
    #
    # tilings = [4, 8, 16, 32]
    #
    # tiles = [4, 8, 16, 32]

    step_sizes = [0.125, 0.06125]

    tilings = [8]

    tiles = [16]

    # THIS IS JUST A TEST
    # We set num_actions in experiment.py

    run_hps = {
        'log_every': 1000,
        'max_eps_steps': float('inf'),
        'max_total_steps': 70000
    }

    all_avg_results = []
    exp_dir = Path(ROOT_DIR, 'experiments')
    timestr = time.strftime("%Y%m%d-%H%M%S")
    print(f"Start experiment for Sarsa(lambda) with Tile Coding at {timestr}")

    env_hps_fname = exp_dir / 'tuning_params.json'
    with open(env_hps_fname, 'r') as f:
        env_hpses = json.load(f)


    # run across all hyperparams
    current_max = None
    current_max_rew = -float('inf')
    for step_size, lam, tiling, tile in product(step_sizes, tilings, tiles):
        agent_hps = {
            'epsilon': 0.01,
            'step_size': step_size,
            'discount': 0.99,
            'iht_size': 3096,
            'lambda': lam,
            'num_tilings': tiling,
            'num_tiles': tile
        }
        print("Experiment on Sarsa Lambda with Tile Coding on hyperparams")
        pp.pprint(agent_hps)
        exp = GeneralizedExperiment(SarsaTCAgent, MountainCarEnv,
                                    agent_hps=agent_hps, env_hpses=env_hpses, run_hps=run_hps,
                                    seeds=[2020])

        exp.run()

        # here we append the average per-episode reward across all 25 tuning
        # environments.
        avg_rew = np.average(exp.all_avg_ep_rews)
        all_avg_results.append((agent_hps, avg_rew))

        if avg_rew > current_max_rew:
            current_max = agent_hps
            current_max_rew = avg_rew

    print(f"Done tuning, best performant agent for Sarsa(lambda) w/ TC is")
    pp.pprint(current_max)
    # Here we need to pick which ones are best

    print("Begin testing")
    # Test here
    test_env_hps_fname = Path(ROOT_DIR, 'experiments', 'testing_params.json')
    with open(test_env_hps_fname, 'r') as f:
        test_env_hpses = json.load(f)

    test_exp = GeneralizedExperiment(SarsaTCAgent, MountainCarEnv,
                                     agent_hps=current_max, env_hpses=test_env_hpses,
                                     run_hps=run_hps, seeds=[2020])

    test_exp.run()

    results = {
        'best_hparams': current_max,
        'avg_ep_rewards': test_exp.all_avg_ep_rews,
        'all_tune_results': all_avg_results
    }

    res_fname = exp_dir / 'sarsa_lambda_tc' / f'sarsa_lambda_tc_results_{timestr}.json'
    print(f"Testing finished. Saving results to {res_fname}")
    with open(res_fname, 'w') as f:
        json.dump(results, f)
