from time import time

import numpy as np
from flatland.envs.rail_env import fast_isclose


def print_timing(label, start_time, end_time):
    print("{:>10.4f}ms".format(1000 * (end_time - start_time)) + "\t" + label)


def check_isclose(nbr=100000):
    s = time()
    for x in range(nbr):
        fast_isclose(x, 0.0, rtol=1e-03)
    e = time()
    print_timing("fast_isclose", start_time=s, end_time=e)

    s = time()
    for x in range(nbr):
        np.isclose(x, 0.0, rtol=1e-03)
    e = time()
    print_timing("np.isclose", start_time=s, end_time=e)


if __name__ == "__main__":
    check_isclose()
