# Same as run.py, but with multiprocessing.

import json
import time
import numpy as np
import multiprocessing as mp
from datetime import timedelta
from pathlib import Path
from itertools import product
from pprint import PrettyPrinter

from grl.generalized_mp_experiment import GeneralizedMPExperiment
from grl.generalized_experiment import GeneralizedExperiment
from grl.agents import SarsaTCAgent
from grl.envs.mountaincar import MountainCarEnv
from definitions import ROOT_DIR


def single_run(agent_hps, env_hpses, run_hps, seeds):
    print("Experiment on Sarsa Lambda with Tile Coding on hyperparams")
    pp.pprint(agent_hps)
    exp = GeneralizedExperiment(SarsaTCAgent, MountainCarEnv,
                                agent_hps=agent_hps, env_hpses=env_hpses, run_hps=run_hps,
                                seeds=seeds)

    exp.run()

    # here we append the average per-episode reward across all 25 tuning
    # environments.
    avg_rew = np.average(exp.all_avg_ep_rews)
    return (agent_hps, avg_rew)


if __name__ == "__main__":
    time_start = time.time()
    pp = PrettyPrinter(indent=4)
    # So here we need to run multiple runs over multiple hyperparams.

    # step_sizes = [1.0, 0.75, 0.5, 0.25, 0.125, 0.06125]
    #
    # tilings = [4, 8, 16, 32]
    #
    # tiles = [4, 8, 16, 32]

    step_sizes = [0.125]

    tilings = [8]

    tiles = [16]

    cheating_tile_range = True

    # THIS IS JUST A TEST
    # We set num_actions in experiment.py

    run_hps = {
        'log_every': 1000,
        'max_eps_steps': float('inf'),
        'max_total_steps': 70000
    }

    exp_dir = Path(ROOT_DIR, 'experiments')
    timestr = time.strftime("%Y%m%d-%H%M%S")
    print(f"Start experiment for Sarsa(lambda) with Tile Coding at {timestr}")

    env_hps_fname = exp_dir / 'tuning_params.json'
    with open(env_hps_fname, 'r') as f:
        env_hpses = json.load(f)


    # run across all hyperparams
    param_list = []
    # manager = mp.Manager()
    # shared_res = manager.list()
    hyperparams = product(step_sizes, tilings, tiles)

    for step_size, tiling, tile in hyperparams:
        agent_hps = {
            'epsilon': 0.01,
            'step_size': step_size,
            'discount': 0.99,
            'iht_size': 4096,
            'num_tilings': tiling,
            'num_tiles': tile,
            'cheating_tile_range': cheating_tile_range
        }
        param_list.append((agent_hps, env_hpses, run_hps, [2020]))


    with mp.Pool() as p:
        res_list = p.starmap(single_run, param_list)

    current_max = None
    current_max_rew = -float('inf')
    for ahps, avg in res_list:
        if current_max_rew < avg:
            current_max = ahps
            current_max_rew = avg

    print(f"Done tuning, best performant agent for Sarsa(lambda) w/ TC is")
    pp.pprint(current_max)
    # Here we need to pick which ones are best

    print("Begin testing")
    # Test here
    test_env_hps_fname = Path(ROOT_DIR, 'experiments', 'testing_params.json')
    with open(test_env_hps_fname, 'r') as f:
        test_env_hpses = json.load(f)

    test_exp = GeneralizedMPExperiment(SarsaTCAgent, MountainCarEnv,
                                     agent_hps=current_max, env_hpses=test_env_hpses,
                                     run_hps=run_hps, seeds=[2020])

    test_exp.run()

    results = {
        'best_hparams': current_max,
        'avg_ep_rewards': test_exp.all_avg_ep_rews,
        'all_tune_results': res_list
    }

    res_fname = exp_dir / 'sarsa_tc' / f'sarsa_tc_results_{timestr}.json'
    print(f"Testing finished. Saving results to {res_fname}")
    with open(res_fname, 'w') as f:
        json.dump(results, f)

    time_end = time.time()
    elapsed = time_end - time_start

    print(f"Time elapsed for experiment: {str(timedelta(seconds=elapsed))}")


