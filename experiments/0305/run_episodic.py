import copy
import json
import math
import time
from pathlib import Path

import fire
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from definitions import ROOT_DIR
from fastprogress.fastprogress import master_bar, progress_bar
from grl.agents.sarsa_lambda_tc import SarsaLambdaTCAgent
from grl.agents.sarsa_nn import DQNAgent as SarsaDQNAgent
from grl.agents.sarsa_nn_tc import SarsaAgent as TCSarsaNNAgent
from grl.agents.sarsa_tc import SarsaAgent as SarsaTCAgent
from grl.envs.mountaincar import MountainCarEnv
from grl.sampling import sample_mountaincar_env
from tqdm import tqdm

num_runs = 1
num_episodes = 300

def run_episode(env, agent):
    is_terminal = False
    sum_of_rewards = 0
    step_count = 0

    obs = env.reset()
    action = agent.agent_start(obs)

    while not is_terminal:
        obs, reward, is_terminal, _ = env.step(action)
        print(agent.steps,end='\r')
        sum_of_rewards += reward
        step_count += 1
        state = obs
        if step_count == math.inf:
            agent.agent_end(reward, state, append_buffer=False)
            break
        elif is_terminal:
            agent.agent_end(reward, state, append_buffer=True)
        else:
            action, _ = agent.agent_step(reward, state)

    return -step_count

agents = {
    "Sarsa_tc": SarsaTCAgent,
    "Tile_Coded_Sarsa_NN": TCSarsaNNAgent,
}

agent_infos = {
    "Sarsa_tc": {
        'step_size': .1,
        'iht_size': 4096,
        'num_tilings': 8,
        'num_tiles': 8,
    },
    "Tile_Coded_Sarsa_NN": {
        'step_size': 1e-4,
        'num_states': 2,
        'iht_size': 4096,
        'num_tilings': 8,
        'num_tiles': 8,
    }
}

metrics = {"ep_rewards": {}, "hyper_params":{}}

def objective(agent_type, hyper_params, num_runs=num_runs, env_idx=0):
    start = time.time()

    ep_rewards = {}

    with open(Path(ROOT_DIR)/'experiments/tuning_params.json', 'r') as f:
        env_infos = json.load(f)

    # env_infos = sample_mountaincar_env(42, 2)
    algorithm = agent_type + '_' + '_'.join([f'{k}_{v}' for k, v in hyper_params.items()])

    mb = master_bar(env_infos)

    for env_name, env_info in enumerate(mb):
        total_steps = 0

        if env_name != env_idx:
            continue
        if env_name not in ep_rewards:
            ep_rewards[env_name] = {}
            print(env_name)
            ep_rewards[env_name][algorithm] = []

        for run in tqdm(range(num_runs)):
            agent = agents[agent_type]()
            env = MountainCarEnv(**env_info)
            print(env_info)
            agent_info = {"num_actions": env.action_space.n, "epsilon": 1, "step_size": 0.1, 'discount': .99, 'iht_size': 4096, "seed": run}
            agent_info.update(agent_infos[agent_type])
            agent_info.update(hyper_params)

            np.random.seed(run)

            agent.agent_init(agent_info)

            rewards = []
            epsilon = .1

            for episode in range(num_episodes):
                print(f"episode {episode}",end='\r')
                agent.epsilon = epsilon
                sum_of_rewards = run_episode(env, agent)
                # epsilon *= 0.99
                rewards.append(sum_of_rewards)
                total_steps -= sum_of_rewards
                if total_steps >= 150000:
                    break

            ep_rewards[env_name].setdefault(algorithm, []).append(rewards)

    end = time.time()
    print(end - start)
    metrics['ep_rewards'] = ep_rewards
    metrics['hyper_params'][algorithm] = hyper_params
    Path(f"{ROOT_DIR}/experiments/0305/metrics/").mkdir(parents=True, exist_ok=True)
    torch.save(metrics, f'{ROOT_DIR}/experiments/0305/metrics/env_{env_idx}_{algorithm}.torch')
    return algorithm, np.mean(metrics["ep_rewards"][env_idx][algorithm]), hyper_params


if __name__ == '__main__':
    fire.Fire(objective)
