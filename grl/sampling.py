import json
from pathlib import Path
import numpy as np
from typing import List

from definitions import ROOT_DIR

def sample_mountaincar_env(seed: int, n: int) -> List[dict]:
    """
    Sample n mountain car environment parameters based on generalizations described in
    https://www.cs.utexas.edu/users/pstone/Papers/bib2html-links/ADPRL11-shimon.pdf
    :param seed: Seed for sampling
    :param n: number of environments to sample
    :return:
    """
    random_state = np.random.RandomState(seed)
    all_envs = []
    for i in range(n):
        sample_p_offset = random_state.uniform(-1.0, 1.0)

        sample_v_offset = random_state.uniform(-1.0, 1.0)

        sample_p_noise_divider = random_state.uniform(5.0, 20.0)

        sample_v_noise_divider = random_state.uniform(5.0, 20.0)

        sample_accel_bias_mean = random_state.uniform(0.8, 1.2)

        # env = MountainCarEnv(accel_bias_mean=sample_accel_bias_mean,
        #                      p_offset=sample_p_offset, v_offset=sample_v_offset,
        #                      p_noise_divider=sample_p_noise_divider,
        #                      v_noise_divider=sample_v_noise_divider)

        # all_envs.append(env)
        all_envs.append({
            'accel_bias_mean': sample_accel_bias_mean,
            'p_offset': sample_p_offset,
            'v_offset': sample_v_offset,
            'p_noise_divider': sample_p_noise_divider,
            'v_noise_divider': sample_v_noise_divider
        })

    return all_envs

if __name__ == "__main__":
    params_file = Path(ROOT_DIR, 'experiments', 'tuning_params.json')

    env_params = sample_mountaincar_env(2020, 100)
    print(f'Saving params to {params_file}')

    with open(params_file, 'w') as f:
        json.dump(env_params, f)


    test_params_file = Path(ROOT_DIR, 'experiments', 'testing_params.json')

    test_env_params = sample_mountaincar_env(2021, 100)
    print(f'Saving test params to {test_params_file}')

    with open(test_params_file, 'w') as f:
        json.dump(test_env_params, f)




