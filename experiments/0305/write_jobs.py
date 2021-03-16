from itertools import product
from pathlib import Path

import numpy as np
from definitions import ROOT_DIR
from fastprogress.fastprogress import master_bar, progress_bar

count = 0

def create_job(agent_type, hyper_params, env):
    global count
    cmd = f"python -m experiments.0305.run_episodic --agent_type=\"{agent_type}\" --env_idx={env} --hyper_params=\"{hyper_params}\""
    jobs_dir = Path(f"{ROOT_DIR}/experiments/0305/jobs/")
    jobs_dir.mkdir(parents=True, exist_ok=True)
    with open(jobs_dir/f"tasks_{count}.sh", 'w') as f:
        f.write(cmd)
    print(count, cmd)
    count += 1

def grid_search(agent_type, param_grid, env):
    """grid search for hyperparameter optimization"""
    param_keys, values = zip(*param_grid.items())

    param_combos = [dict(zip(param_keys, combo)) for combo in product(*values)]

    mb = master_bar(param_combos)

    for i, hyper_params in enumerate(mb):
        print(hyper_params)
        create_job(agent_type, hyper_params, env)

agents = {
    # "Sarsa_tc": None,
    "Tile_Coded_Sarsa_NN": None,
}

def get_lr(b=1e-2, a=2, n=5):
    return list(b/a**np.array(list(range(0, n))))

params_to_search = {
    "Sarsa_tc": {
        'step_size': [1.0, 0.5, 0.25, 0.125, 0.06125],
        'num_tilings': [4, 8, 16, 32],
        'num_tiles': [4, 8, 16, 32],
    },
    "Tile_Coded_Sarsa_NN": {
        'step_size': get_lr(),
        'num_tilings': [4, 8, 16],
        'num_tiles': [4, 8, 16, 32],
    }
}


if __name__ == "__main__":
    for env in range(5):
        for agent_type in master_bar(list(agents.keys())):
            print(env, agent_type)
            grid_search(agent_type, params_to_search[agent_type], env)
            print('Jobs: sbatch --array={}-{} ./experiments/0305/jobs/run_cpu.sh'.format(0, count-1))
