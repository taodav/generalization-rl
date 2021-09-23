import json
import time
import numpy as np
import argparse
from pathlib import Path
from itertools import product
from pprint import PrettyPrinter

from grl.generalized_experiment import GeneralizedExperiment
from grl.agents import DQNAgent
# from grl.agents import SarsaTCAgent
from grl.envs.mountaincar import MountainCarEnv
from definitions import ROOT_DIR

def get_lr(b=1e-2, a=2, n=5):
    return list(b/a**np.array(list(range(0, n))))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-target", help="Don't use target networks",
                        action="store_true")
    parser.add_argument("--no-replay", help="Don't use experience replay",
                        action="store_true")
    args = parser.parse_args()


    pp = PrettyPrinter(indent=4)
    # So here we need to run multiple runs over multiple hyperparams.

    # step_sizes = [1.0, 0.75, 0.5, 0.25, 0.125, 0.06125]
    #
    # tilings = [8, 16, 32]
    #
    # tiles = [8, 16, 32]

    max_replay_sizes = [10000]

    update_target_intervals = [10000]

    step_sizes = get_lr()[-3:-2]


    # THIS IS JUST A TEST
    # We set num_actions in experiment.py

    run_hps = {
        'log_every': 1000,
        'max_eps_steps': float('inf'),
        'max_total_steps': 150000
    }

    all_avg_results = []
    exp_dir = Path(ROOT_DIR, 'experiments')
    timestr = time.strftime("%Y%m%d-%H%M%S")
    print(f"Start experiment for DQN at {timestr}")

    env_hps_fname = exp_dir / 'tuning_params.json'
    with open(env_hps_fname, 'r') as f:
        env_hpses = json.load(f)


    # run across all hyperparams
    current_max = None
    current_max_rew = -float('inf')
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
        print("Experiment on DQN on hyperparams")
        pp.pprint(agent_hps)
        exp = GeneralizedExperiment(DQNAgent, MountainCarEnv,
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

    print(f"Done tuning, best performant agent for DQN is")
    pp.pprint(current_max)
    # Here we need to pick which ones are best

    print("Begin testing")
    # Test here
    test_env_hps_fname = Path(ROOT_DIR, 'experiments', 'testing_params.json')
    with open(test_env_hps_fname, 'r') as f:
        test_env_hpses = json.load(f)

    test_exp = GeneralizedExperiment(DQNAgent, MountainCarEnv,
                                     agent_hps=current_max, env_hpses=test_env_hpses,
                                     run_hps=run_hps, seeds=[2020])

    test_exp.run()

    results = {
        'best_hparams': current_max,
        'avg_ep_rewards': test_exp.all_avg_ep_rews,
        'all_tune_results': all_avg_results
    }

    fname = "dqn"

    if args.no_target:
        fname += "_no_target"

    if args.no_replay:
        fname += "_no_replay"

    res_fname = exp_dir / 'dqn' / f'{fname}_results_{timestr}.json'
    print(f"Testing finished. Saving results to {res_fname}")
    with open(res_fname, 'w') as f:
        json.dump(results, f)
