from functools import partial
from math import pi, sin, sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def do_show(time: list, z: list, v: list):
    fig, ax = plt.subplots()
    ax.plot(time, z, label="z-position")
    ax.plot(time, v, label="z-speed")
    ax.legend()
    plt.show()


def force(t: float, ampl: float = 1.0, omega: float = 0.1):
    return np.array((0, 0, ampl * sin(omega * t)), float)


def test_oscillator_class(show):
    """Test the Oscillator class in isolation.

    The first four lines are necessary to ensure that the Oscillator class can be accessed:
    If pytest is run from the command line, the current directory is the package root,
    but when it is run from the editor (__main__) it is run from /tests/.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).parent / "examples"))
    from oscillator import HarmonicOscillator

    osc = HarmonicOscillator(k=1.0, c=0.1, m=1.0)
    osc.x[2] = 1.0
    times = []
    z = []
    v = []
    _f = partial(force, ampl=1.0, omega=0.1)
    dt = 0.01
    time = 0
    print("Period", 2 * pi / sqrt(osc.k / osc.m))
    for _ in range(10000):
        osc.f = _f(time)
        osc.do_step(time, dt)
        times.append(time)
        z.append(osc.x[2])
        v.append(osc.v[2])
        time += dt

    if show:
        do_show(times, z, v)


if __name__ == "__main__":
    # retcode = pytest.main(["-rA", "-v", "--rootdir", "../", "--show", "False", __file__])
    # assert retcode == 0, f"Non-zero return code {retcode}"
    test_oscillator_class(show=True)
