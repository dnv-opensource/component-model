from functools import partial
from math import atan2, cos, exp, pi, sin, sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest


def do_show(
    time: list,
    z: list,
    v: list,
    compare1: list | None = None,
    compare2: list | None = None,
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


def force_xv(
    dim: int = 3,
    t: float | None = None,
    x: np.ndarray | None = None,
    v: np.ndarray | None = None,
    dt: float | None = None,
    const: float | None = None,
    d_omega: float = 0.0,
    ampl: float | None = None,
    omega: float = 0.1,
    ampl_x: float | None = None,
    ampl_v: float | None = None,
):
    """Use a force which is dependent on position and/or velocity. Time is ignored"""
    force = np.array((0,) * dim, float)
    if isinstance(const, float):
        force += np.array((const,) * dim, float)
    if t is not None and ampl is not None:
        if d_omega == 0.0:
            force += np.array((0, 0, ampl * sin(omega * t)), float)  # fixed frequency
        else:
            force += np.array((0, 0, ampl * sin((omega + d_omega * t) * t)), float)  # frequency sweep
    if isinstance(ampl_x, float):
        force += ampl_x * x  # type: ignore[operator] ##it is definitely float* ndarray
    if isinstance(ampl_v, float):
        force += ampl_v * v  # type: ignore[operator] ##it is definitely float* ndarray
    return force


def forced_oscillator(
    t: float, k: float, c: float, m: float, a: float = 0.0, wf: float = 0.1, x0: float = 1.0, v0: float = 0.0
):
    """Calculates the expected (analytic) position and speed of a harmonic oscillator (in one dimension)
    with the given parameter setting.

    Args:
        t (float): time
        k,c,m (float): harmonic oscillator parameters
        a,wf (float): sinusoidal force parameters (amplitude and angular frequency)
        x0, v0 (float): start values for harmonic oscillator (force has fixed start values)
    """
    from math import atan2, sin, sqrt

    w0 = sqrt(k / m)  # omega0
    b = c / (2 * m)  # beta
    if a != 0:
        assert x0 == 0 and v0 == 0, "Checking of forced oscillations is only implemented for x0=0 and v0=0"
        A = a / sqrt(((w0**2 - wf**2) ** 2 + (2 * b * wf) ** 2))
        d = atan2(2 * b * wf, w0**2 - wf**2)  # phase angle in equilibrium
        x0 = A * sin(d)
        v0 = -A * wf * cos(d)
        x_e = A * sin(wf * t - d)
        v_e = A * wf * cos(wf * t - d)
        # return x_e, v_e
    else:
        x_e, v_e = (0.0, 0.0)

    if w0 - b > 1e-10:  # damped oscillation
        w1 = sqrt(w0**2 - b**2)  # angular frequency of oscillation
        x = exp(-b * t) * (x0 * cos(w1 * t) + (x0 * b + v0) / w1 * sin(w1 * t))
        v = exp(-b * t) * (v0 * cos(w1 * t) - (w1 * x0 + b**2 / w1 * x0 + b / w1 * v0) * sin(w1 * t))
    elif abs(w0 - b) < 1e-10:  # critically damped oscillation
        x = ((v0 + b * x0) * t + x0) * exp(-b * t)
        v = -b * (v0 + b * x0) * t * exp(-b * t)
    else:  # over-damped oscillation
        w1_ = sqrt(b**2 - w0**2)
        _b1 = 2 * b - w1_
        _b2 = 2 * b + w1_
        x = ((b + w1_) * x0 + v0) / 2 / w1_ * exp(-(b - w1_) * t) - ((b - w1_) * x0 + v0) / 2 / w1_ * exp(
            -(b + w1_) * t
        )
        v = -((b + w1_) * x0 + v0) / 2 / w1_ * (b - w1_) * exp(-(b - w1_) * t) + ((b - w1_) * x0 + v0) / 2 / w1_ * (
            b + w1_
        ) * exp(-(b + w1_) * t)
    if a != 0:
        return (x + x_e, v + v_e)
    else:
        return (x, v)


def run_oscillation_z(
    k: float,
    c: float,
    m: float,
    ampl: float,
    omega: float,
    x0: float = 1.0,
    v0: float = 0.0,
    dt: float = 0.01,
    end: float = 30.0,
    tol: float = 1e-3,
):
    """Run the oscillator with the given settings for the given time (only z-direction activated)
    and return the oscillator object and the time series for z-position and z-velocity."""

    from examples.oscillator_xd import Force, OscillatorXD

    _f = partial(force_xv, dim=3, ampl=ampl, omega=omega)
    _force = Force(3, _f)
    osc = OscillatorXD(dim=3, k=(1.0, 1.0, k), c=(0.0, 0.0, c), m=m, tolerance=tol, force=_force)
    osc.x[2] = x0  # set initial z value
    osc.v[2] = v0  # set initial z-speed
    times, z, v = [], [], []
    time = 0.0
    while time < end:
        times.append(time)
        z.append(osc.x[2])
        v.append(osc.v[2])
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

    from examples.oscillator_xd import Force, OscillatorXD

    _f = partial(force_xv, dim=3, ampl=ampl, omega=0.0, d_omega=d_omega)
    _force = Force(3, _f)
    osc = OscillatorXD(dim=3, k=(1.0, 1.0, k), c=(0.0, 0.0, c), m=m, tolerance=tol, force=_force)
    osc.x[2] = x0  # set initial z value
    osc.v[2] = v0  # set initial z-speed
    times, z, v, f = [], [], [], []
    time = 0.0
    while time < end:
        times.append(time)
        z.append(osc.x[2])
        v.append(osc.v[2])
        osc.do_step(time, dt)
        f.append(_f(t=time)[2])
        time += dt

    return (osc, times, z, v, f)


def test_oscillator_class(show):
    """Test the Oscillator class in isolation.
    Such tests are strongly recommended before compiling the model into an FMU.

    With respect to `wiki <https://en.wikipedia.org/wiki/Oscillation>`_ our parameters are:
    b = c, k=k => beta = c/(2m), w0 = sqrt(k/m) => w1 = sqrt(beta^2 - w0^2) = sqrt( c^2/4/m^2 - k/m)
    """
    test_cases: list[tuple[float, float, float, float, float, float, str]] = [
        # k    c    m    a    w    x0   description
        (1.0, 0.0, 1.0, 0.0, 0.1, 1.0, "Oscillator without damping and force"),
        (1.0, 0.2, 1.0, 0.0, 0.1, 1.0, "Oscillator include damping"),
        (1.0, 2.0, 1.0, 0.0, 0.1, 1.0, "Oscillator critically damped"),
        (1.0, 5.0, 1.0, 0.0, 0.1, 1.0, "Oscillator over-damped"),
        (1.0, 0.2, 1.0, 1.0, 0.5, 0.0, "Forced oscillation. Less than resonance freq"),
        (1.0, 0.2, 1.0, 1.0, 1.0, 0.0, "Forced oscillation. Damped. Resonant"),
        (1.0, 0.2, 1.0, 1.0, 2.0, 0.0, "Forced oscillation. Damped. Above resonance freq"),
        (1.0, 2.0, 1.0, 1.0, 1.0, 0.0, "Forced oscillation. Crit. damped. Resonant"),
        (1.0, 5.0, 1.0, 1.0, 1.0, 0.0, "Forced oscillation. Over-damped. Resonant"),
    ]
    tol = 1e-3  # tolerance for simulation
    for k, c, m, a, w, x0, msg in test_cases:
        print(f"{msg}: k={k}, c={c}, m={m}, a={a}, wf={w}", end="")
        osc, t, z, v = run_oscillation_z(k=k, c=c, m=m, ampl=a, omega=w, x0=x0, tol=tol)
        if c < 2.0:  # only if damping is small enough
            cp = 2.0 * pi / sqrt(k / m - (c / 2.0 / m) ** 2)
            assert abs(osc.period[2] - cp) < 1e-12, f"Period[{2}]: {osc.period[2]} != {cp}. {osc.w2}, {osc.gam}"
        x_expect, v_expect = [], []
        for ti in t:
            _x, _v = forced_oscillator(ti, k, c, m, a, w, x0=x0)
            x_expect.append(_x)
            v_expect.append(_v)
        if show:
            do_show(t, z, v, x_expect, v_expect)
        emax = 0.0
        for i, ti in enumerate(t):
            assert abs(z[i] - x_expect[i]) < 50 * tol, f"@{ti}: z={z[i]} != {x_expect[i]}"
            assert abs(v[i] - v_expect[i]) < 50 * tol, f"@{ti}: v={v[i]} != {v_expect[i]}"
            emax = max(emax, abs(z[i] - x_expect[i]), abs(v[i] - v_expect[i]))
        print(f". Max absolute error: {emax}")


def test_2d(show):
    from examples.oscillator_xd import OscillatorXD

    def run_2d(
        x0: tuple[float, float, float],
        v0: tuple[float, float, float],
        k: tuple[float, float, float] = (1.0, 1.0, 1.0),
        c: tuple[float, float, float] = (0.0, 0.0, 0.0),
        end: float = 100.0,
        dt: float = 0.01,
        tolerance: float = 1e-5,
    ):
        osc = OscillatorXD(dim=3, k=k, c=c, tolerance=tolerance)
        osc.x[:3] = np.array(x0, float)  # set initial 3D position
        osc.x[3:] = np.array(v0, float)  # set initial 3D speed
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
    assert np.allclose(osc.period, (2 * np.pi, 2 * np.pi, 2 * np.pi)), f"Found {osc.period}"
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
    osc, times0, z0, v0, f0 = sweep_oscillation_z(
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
        fig, ax = plt.subplots()
        ax.plot(freq, z0, label="z0(t)")
        ax.plot(freq, v0, label="v0(t)")
        # ax.plot(freq, f0, label="F0(t)")
        ax.legend()
        plt.show()

    osc, times, z, v, f = sweep_oscillation_z(
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

        assert abs(z0[i0] - z[i]) < 4e-2, f"Time {t}. Found {z0[i0]} != {z[i]}"
        assert abs(v0[i0] - v[i]) < 4e-2, f"Time {t}. Found {v0[i0]} != {v[i]}"

    if show:
        fig, ax = plt.subplots()
        ax.plot(times0, z0, label="z0(t)")
        ax.plot(times, z, label="z(t)")
        ax.legend()
        plt.show()


def test_forced_xv(show: bool = False):
    from examples.oscillator_xd import Force, OscillatorXD

    def do_scenario(
        k: float = 1.0,
        c: float = 0.0,
        ax: float = 0.0,
        av: float = 0.0,
        const: float | None = None,
        v0: float = 1.0,
        show=show,
        title: str = "Scenario",
    ):
        _f = partial(force_xv, dim=6, ampl_x=ax, ampl_v=av, const=const)
        _force = Force(6, _f)
        osc = OscillatorXD(dim=6, k=(k,) * 6, c=(c,) * 6, m=1.0, tolerance=1e-3, force=_force)
        osc.v[2] = v0  # set initial z-speed
        times, z, v = [], [], []
        time = 0.0
        dt = 0.1
        while time < 50.0:
            times.append(time)
            z.append(osc.x[2])
            v.append(osc.v[2])
            osc.do_step(time, dt)
            time += dt

        if show:
            fig, axis = plt.subplots()
            axis.plot(times, z, label="z0(t)")
            axis.plot(times, v, label="v0(t)")
            plt.legend()
            axis.set_title(title)
            plt.show()
        return (z, v)

    z, v = do_scenario(const=-10.0, v0=0.0, show=False, title="Constant force -10.0")
    assert abs(sum(v) / len(v)) < 0.01, "Velocity still around +-0"
    assert abs(sum(z) / len(z) + 10.0) < 0.06, "New equilibrium with constant force"
    return
    z0, v0 = do_scenario(show=False, title="Basic oscillator with start velocity 1.0")
    z, v = do_scenario(k=0, ax=-1.0, show=False, title="Spring constant 0.0, replaced with force(position)")
    assert np.allclose(z0, z), "Same effect as change in frequency when force dependent on position."
    assert np.allclose(v0, v), "Same effect as change in frequency when force dependent on position."
    z0, v0 = do_scenario(c=0.1, show=False, title="Basic oscillator with damping=0.1 and start veocity=1.0")
    z, v = do_scenario(c=0.0, av=-0.1, show=False, title="Daming replaced by force(velocity)")
    assert np.allclose(z0, z), "Same effect as change in damping when force dependent on velocity."
    assert np.allclose(v0, v), "Same effect as change in damping when force dependent on velocity."


if __name__ == "__main__":
    retcode = pytest.main(["-rA", "-v", "--rootdir", "../", "--show", "False", __file__])
    assert retcode == 0, f"Non-zero return code {retcode}"
    import os

    os.chdir(Path(__file__).parent.absolute() / "test_working_directory")
    # test_oscillator_class(show=True)
    # test_2d(show=True)
    # test_sweep_oscillator(show=True)
    # test_forced_xv(show=True)
