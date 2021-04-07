import json
import time
import numpy as np
from torch.multiprocessing import Pool, set_start_method
try:
     set_start_method('spawn')
except RuntimeError:
    pass
import argparse
from pathlib import Path
from itertools import product
from pprint import PrettyPrinter
from datetime import timedelta

from grl.generalized_experiment import GeneralizedExperiment
from grl.generalized_mp_experiment import GeneralizedMPExperiment
from grl.agents import DQNAgent
from grl.envs.mountaincar import MountainCarEnv
from definitions import ROOT_DIR

def get_lr(b=1e-2, a=2, n=5):
    return list(b/a**np.array(list(range(0, n))))

def single_run(agent_hps, env_hpses, run_hps, seeds):
    print("Experiment on DQN on hyperparams")
    pp = PrettyPrinter(indent=4)
    pp.pprint(agent_hps)
    exp = GeneralizedExperiment(DQNAgent, MountainCarEnv,
                                agent_hps=agent_hps, env_hpses=env_hpses, run_hps=run_hps,
                                seeds=seeds)

    exp.run()

    # here we append the average per-episode reward across all 25 tuning
    # environments.
    avg_rew = np.average(exp.all_avg_ep_rews)
    return (agent_hps, avg_rew)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-target", help="Don't use target networks",
                        action="store_true")
    parser.add_argument("--no-replay", help="Don't use experience replay",
                        action="store_true")
    args = parser.parse_args()

    time_start = time.time()
    pp = PrettyPrinter(indent=4)
    # So here we need to run multiple runs over multiple hyperparams.

    max_replay_sizes = [100, 10000, 70000]

    update_target_intervals = [100, 10000, 70000]

    step_sizes = get_lr()


    # max_replay_sizes = [10000]
    #
    # update_target_intervals = [10000]
    #
    # step_sizes = get_lr()[-3:-2]


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
    param_list = []
    for step_size, replay_size, update_target in product(step_sizes, max_replay_sizes, update_target_intervals):
        agent_hps = {
            'batch_size': 32,
            'epsilon': 0.01,
            'step_size': step_size,
            'discount': 0.99,
            'max_replay_size': replay_size,
            'update_target_interval': update_target,
            'use_replay': not args.no_replay,
            'use_target': not args.no_target
        }

        param_list.append((agent_hps, env_hpses, run_hps, [2020]))

    with Pool() as p:
        res_list = p.starmap(single_run, param_list)

    current_max = None
    current_max_rew = -float('inf')
    for ahps, avg in res_list:
        if avg > current_max_rew:
            current_max = ahps
            current_max_rew = avg
    # current_max = param_list[0][0]


    print(f"Done tuning, best performant agent for Sarsa(lambda) w/ TC is")
    pp.pprint(current_max)
    # Here we need to pick which ones are best

    print("Begin testing")
    # Test here
    test_env_hps_fname = Path(ROOT_DIR, 'experiments', 'testing_params.json')
    with open(test_env_hps_fname, 'r') as f:
        test_env_hpses = json.load(f)

    test_exp = GeneralizedMPExperiment(DQNAgent, MountainCarEnv,
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

    time_end = time.time()
    elapsed = time_end - time_start

    print(f"Time elapsed for experiment: {str(timedelta(seconds=elapsed))}")
