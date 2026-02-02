import logging
from functools import partial
from math import atan2, pi, sin, sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from component_model.analytic import ForcedOscillator1D  # , sine_fit

logger = logging.getLogger(__name__)


def arrays_equal(res: tuple[float, ...] | list[float], expected: tuple[float, ...] | list[float], eps: float = 1e-7):
    assert len(res) == len(expected), (
        f"Tuples of different lengths cannot be equal. Found {len(res)} != {len(expected)}"
    )
    for i, (x, y) in enumerate(zip(res, expected, strict=False)):
        assert abs(x - y) < eps, f"Element {i} not nearly equal in {x}, {y}"
    return True


def do_show(
    time: list[float] | np.ndarray,
    z: list[float] | np.ndarray,
    v: list[float] | np.ndarray,
    compare1: list[float] | np.ndarray | None = None,
    compare2: list[float] | np.ndarray | None = None,
    z_label: str = "z-position",
    v_label: str = "z-speed",
):
    fig, ax = plt.subplots()
    ax.plot(time, z, label=z_label)
    ax.plot(time, v, label=v_label)
    if compare1 is not None:
        ax.plot(time, compare1, label="compare1")
    if compare2 is not None:
        ax.plot(time, compare2, label="compare2")
    ax.legend()
    plt.show()


def force(t: float, ampl: float = 1.0, omega: float = 0.1, d_omega: float = 0.0):
    if d_omega == 0.0:
        return np.array((0, 0, ampl * sin(omega * t)), float)  # fixed frequency
    else:
        return np.array((0, 0, ampl * sin((omega + d_omega * t) * t)), float)  # frequency sweep


def run_oscillation_z(
    k: float,
    c: float,
    m: float,
    ampl: float,
    omega: float,
    x0: float = 0.0,
    v0: float = 1.0,
    dt: float = 0.01,
    end: float = 50.0,
    tol: float = 1e-3,
):
    """Run the oscillator with the given settings for the given time (only z-direction activated)
    and return the oscillator object and the time series for z-position and z-velocity."""

    from examples.oscillator import Oscillator

    osc = Oscillator(k=(1.0, 1.0, k), c=(0.0, 0.0, c), m=m, tolerance=tol)
    osc.x[2] = x0  # set initial z value
    osc.v[2] = v0  # set initial z-speed
    times, z, v = [], [], []
    _f = partial(force, ampl=ampl, omega=omega)
    time = 0.0
    while time < end:
        times.append(time)
        z.append(osc.x[2])
        v.append(osc.v[2])
        osc.f = _f(time)
        osc.do_step(time, dt)
        time += dt

    return (osc, times, z, v)


def sweep_oscillation_z(
    k: float,
    c: float,
    m: float,
    ampl: float,
    d_omega: float,
    x0: float = 1.0,
    v0: float = 0.0,
    dt: float = 0.01,
    end: float = 30.0,
    tol: float = 1e-3,
):
    """Run the oscillator with the given settings
    with linearly increasing force frequency
    for the given time (only z-direction activated)
    and return the oscillator object and the time series for z-position and z-velocity."""

    from examples.oscillator import Oscillator

    f_func = f_func = partial(force, ampl=ampl, omega=0.0, d_omega=d_omega)
    osc = Oscillator(k=(1.0, 1.0, k), c=(0.0, 0.0, c), m=m, tolerance=tol, f_func=f_func)
    a_osc = ForcedOscillator1D(k=k, c=c, m=m, a=ampl, wf=1, x0=x0, dx0=v0)
    osc.x[2] = x0  # set initial z value
    osc.v[2] = v0  # set initial z-speed
    times, z, v, f, _z, _v = [], [], [], [], [], []
    time = 0.0
    while time < end:
        times.append(time)
        z.append(osc.x[2])
        v.append(osc.v[2])
        osc.do_step(time, dt)
        f.append(f_func(time)[2])
        # analytic solution for comparison
        a_osc.coefficients(wf=d_omega * time)
        zz, vv = a_osc.calc(time)
        _z.append(zz)
        _v.append(vv)
        time += dt

    return (osc, times, z, v, f, _z, _v)


def test_oscillator_class(show: bool = False):
    """Test the Oscillator class in isolation.
    Such tests are strongly recommended before compiling the model into an FMU.

    With respect to `wiki <https://en.wikipedia.org/wiki/Oscillation>`_ our parameters are:
    b = c, k=k => beta = c/(2m), w0 = sqrt(k/m) => w1 = sqrt(beta^2 - w0^2) = sqrt( c^2/4/m^2 - k/m)
    """
    test_cases: list[tuple[float, float, float, float, float, str]] = [
        # k    c    m    a    w   description
        (1.0, 0.0, 1.0, 0.0, 0.1, "Oscillator without damping and force"),
        (1.0, 0.2, 1.0, 0.0, 0.1, "Oscillator include damping"),
        (1.0, 2.0, 1.0, 0.0, 0.1, "Oscillator critically damped"),
        (1.0, 5.0, 1.0, 0.0, 0.1, "Oscillator over-damped"),
        (1.0, 0.2, 1.0, 1.0, 0.5, "Forced oscillation. Less than resonance freq"),
        (1.0, 0.2, 1.0, 1.0, 1.0, "Forced oscillation. Damped. Resonant"),
        (1.0, 0.2, 1.0, 1.0, 2.0, "Forced oscillation. Damped. Above resonance freq"),
        (1.0, 2.0, 1.0, 1.0, 1.0, "Forced oscillation. Crit. damped. Resonant"),
        (1.0, 5.0, 1.0, 1.0, 1.0, "Forced oscillation. Over-damped. Resonant"),
    ]
    tol = 1e-3  # tolerance for simulation
    for k, c, m, a, w, msg in test_cases:
        a_osc = ForcedOscillator1D(k=k, c=c, m=m, a=a, wf=w, x0=0, dx0=1.0 if a == 0.0 else 0.0)
        if a != 0.0:
            a_osc.coefficients(d0=2 * a_osc.d)
        logger.info(f"{msg}. k:{k}, c:{c}, a:{a}, w:{w}, x0:{a_osc.x0}, v:{a_osc.dx0}")
        osc, t, z, v = run_oscillation_z(k=k, c=c, m=m, ampl=a, omega=w, v0=1.0 if a == 0 else 0.0, tol=tol)
        # a, w, phi = sine_fit(t, z)
        # print(f"{a}* sin({w}* t + {phi})")
        # print( f"x0:{z[0]} - {a_osc.x0}, v0:{v[0]} - {a_osc.dx0}, force: {a_osc.a}, {a_osc.wf}, {a_osc.phi}")
        if c < 2.0:  # only if damping is small enough
            cp = 2.0 * pi / sqrt(k / m - (c / 2.0 / m) ** 2)
            assert abs(osc.period[2] - cp) < 1e-12, f"Period: {osc.period} != {2 * pi}"
        t = np.array(t, float)
        x_expect, v_expect = a_osc.calc(t)
        if show:
            do_show(t, z, v, x_expect, v_expect)
        emax = 0.0
        for i, ti in enumerate(t):
            assert abs(z[i] - x_expect[i]) < 50 * tol, f"@{ti}: z={z[i]} != {x_expect[i]}"
            assert abs(v[i] - v_expect[i]) < 50 * tol, f"@{ti}: v={v[i]} != {v_expect[i]}"
            emax = max(emax, abs(z[i] - x_expect[i]), abs(v[i] - v_expect[i]))
        print(f". Max absolute error: {emax}")


def test_2d(show: bool = False):
    from examples.oscillator import Oscillator

    def run_2d(
        x0: tuple[float, float, float],
        v0: tuple[float, float, float],
        k: tuple[float, float, float] = (1.0, 1.0, 1.0),
        c: tuple[float, float, float] = (0.0, 0.0, 0.0),
        end: float = 100.0,
        dt: float = 0.01,
        tolerance: float = 1e-5,
    ):
        osc = Oscillator(k=k, c=c, tolerance=tolerance)
        osc.x = np.array(x0, float)  # set initial 3D position
        osc.v = np.array(v0, float)  # set initial 3D speed
        x, y = [], []
        t0 = 0.0
        for time in np.linspace(dt, end, int(end / dt), endpoint=True):
            x.append(osc.x[0])
            y.append(osc.x[1])
            osc.do_step(time, time - t0)
            t0 = time
        x.append(osc.x[0])
        y.append(osc.x[1])

        return (osc, x, y)

    def show_2d(x: list[float], y: list[float]):
        fig, ax = plt.subplots()
        ax.plot(x, y, label="x-y")
        ax.legend()
        plt.show()

    def area(x: list[float], y: list[float]):
        """Calculate the area within the curve."""
        angle0 = 0.0
        area = 0.0
        anglesum = 0.0
        for _x, _y in zip(x, y, strict=False):
            angle = atan2(_y, _x)
            dangle = min((2 * np.pi) - abs(angle0 - angle), abs(angle0 - angle))
            area += (_x**2 + _y**2) * dangle / 2
            anglesum += dangle
            angle0 = angle
        return area

    osc, x, y = run_2d(x0=(1.0, 0.0, 0.0), v0=(0.0, 1.0, 0.0), end=2 * np.pi, tolerance=1e-5)
    assert arrays_equal(osc.period, (2 * np.pi, 2 * np.pi, 2 * np.pi)), f"Found {osc.period}"
    assert abs(area(x, y) - np.pi) < 1e-10, f"Found area {area(x, y)}"
    if show:
        show_2d(x, y)
    for _x, _y in zip(x, y, strict=False):
        assert abs(_x**2 + _y**2 - 1.0) < 1e-10, f"Found {_x}**2 + {_y}**2 = {_x**2 + _y**2} != 1.0"

    osc, x, y = run_2d(x0=(1.0, 0.0, 0.0), v0=(0.0, 1.0, 0.0), c=(0.5, 0.5, 0), end=10 * np.pi)
    assert (area(x, y) - 0.9977641389836932) < 1e-15

    if show:
        show_2d(x, y)

    osc, x, y = run_2d(x0=(1.0, 0.0, 0.0), v0=(0.0, 1.0, 0.0), k=(1.0, 1.0 / 16, 0), end=10 * np.pi)
    assert abs(x[-1] - 1.0) < 1e-12, f"Found {x[-1]}"
    assert abs(y[-1] - 4.0) < 1e-12, f"Found {y[-1]}"
    osc, x, y = run_2d(x0=(1.0, 0.0, 0.0), v0=(0.0, 1.0, 0.0), k=(1.0, 1.0 / 15.8, 0), end=20 * np.pi)
    if show:
        show_2d(x, y)


def test_sweep_oscillator(show: bool = False):
    """A forced oscillator where the force frequency is changed linearly as d_omega*time.
    The test demonstrates that a monolithic simulation provides accurate results in all ranges of the force frequency.
    Co-simulating the oscillator and the force, this does not work.
    """
    osc, times0, z0, v0, f0, _z, _v = sweep_oscillation_z(
        k=1.0,
        c=0.1,
        m=1.0,
        ampl=1.0,
        d_omega=0.1,
        x0=0.0,
        v0=0.0,
        dt=0.1,  # 'ground truth', small dt
        end=100.0,
        tol=1e-3,
    )
    with open(Path.cwd() / "oscillator_sweep0.dat", "w") as fp:
        for i in range(len(times0)):
            fp.write(f"{times0[i]}\t{z0[i]}\t{v0[i]}\t{f0[i]}\n")

    if show:
        freq = [0.1 * t / 2 / np.pi for t in times0]
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.plot(freq, z0, label="z0(t)")
        ax1.plot(freq, v0, label="v0(t)")
        ax2.plot(freq, _z, label="z(t)")
        # ax2.plot(freq, _v, label="v(t)")
        # ax.plot(freq, f0, label="F0(t)")
        ax1.legend()
        ax2.legend()
        plt.show()

    osc, times, z, v, f, _z, _v = sweep_oscillation_z(
        k=1.0,
        c=0.1,
        m=1.0,
        ampl=1.0,
        d_omega=0.1,
        x0=0.0,
        v0=0.0,
        dt=1,  # dt similar to resonance frequency
        end=100.0,
        tol=1e-3,
    )
    i0 = 0
    for i in range(len(times)):  # demonstrate that the results are accurate, even if dt is large
        t = times[i]
        while abs(times0[i0] - t) > 1e-10:
            i0 += 1
            assert times0[i0] - t < 0.1, f"Time entry for time {t} not found in times0"

        assert abs(z0[i0] - z[i]) < 2e-2, f"Time {t}. Found {z0[i0]} != {z[i]}"
        assert abs(v0[i0] - v[i]) < 2e-2, f"Time {t}. Found {v0[i0]} != {v[i]}"

    if show:
        fig, ax = plt.subplots()
        ax.plot(times0, z0, label="z0(t)")
        ax.plot(times, z, label="z(t)")
        ax.legend()
        plt.show()


if __name__ == "__main__":
    retcode = pytest.main(["-rA", "-v", "--rootdir", "../", "--show", "False", __file__])
    assert retcode == 0, f"Non-zero return code {retcode}"
    import os

    os.chdir(Path(__file__).parent.absolute() / "test_working_directory")
    # test_oscillator_class(show=True)
    # test_2d(show=True)
    # test_sweep_oscillator(show=True)
