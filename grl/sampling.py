import numpy as np
from typing import Callable, List

from grl.envs.mountaincar import MountainCarEnv


def sample_mountaincar_env(seed: int, n: int) -> List[MountainCarEnv]:
    """
    Sample n mountain car environments based on generalizations described in
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

        env = MountainCarEnv(accel_bias_mean=sample_accel_bias_mean,
                             p_offset=sample_p_offset, v_offset=sample_v_offset,
                             p_noise_divider=sample_p_noise_divider,
                             v_noise_divider=sample_v_noise_divider)

        all_envs.append(env)

    return all_envs



