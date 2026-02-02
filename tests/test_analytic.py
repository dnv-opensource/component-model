import logging

import matplotlib.pyplot as plt
import numpy as np
import pytest

from component_model.analytic import ForcedOscillator1D, sine_fit

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def do_show(
    times: list[float] | np.ndarray,
    traces: dict[str, list[list[float]] | np.ndarray],
    selection: dict[str, int] | None = None,
    title: str = "",
):
    """Plot selected traces."""
    fig, (ax1, ax2) = plt.subplots(1, 2)
    for label, trace in traces.items():
        if selection is None:  # all in first subplot
            _ = ax1.plot(times, trace, label=label)
        else:
            if label in selection:
                if selection[label] == 1:
                    _ = ax1.plot(times, trace, label=label)
                elif selection[label] == 2:
                    _ = ax2.plot(times, trace, label=label)
    _ = ax1.legend()
    _ = ax2.legend()
    plt.title(title, loc="left")
    plt.show()


def test_amplitude_omega_phase(show: bool = False):
    times = np.linspace(0.0, 50, 101)
    trace = 9.9 * np.sin(1.5 * times)
    a, w, phi = sine_fit(times, trace)
    if show:
        do_show(times, {"trace": trace, "fit": a * np.sin(w * times + phi)})
    assert np.allclose((a, w, phi), (9.9, 1.5, 0.0)), f"Found a:{a}, w:{w}, phi:{phi}"

    # rotate the phase angle one whole round in 1 degree steps
    for angle in np.linspace(-np.pi + np.radians(1.0), np.pi, 360):
        trace = 9.9 * np.sin(1.5 * times + angle)
        a, w, phi = sine_fit(times, trace)
        if not np.allclose((a, w, phi), (9.9, 1.5, angle)):
            if show:
                do_show(times, {"trace": trace, "fit": a * np.sin(w * times + phi)}, title=f"{angle} != {phi}")
            assert np.allclose((a, w, phi), (9.9, 1.5, angle)), f"@{angle}. Found a:{a}, w:{w}, phi:{phi}"

    trace = 9.9 * np.exp(-0.1 * times) * np.sin(1.5 * times + np.pi / 4)
    a, w, phi = sine_fit(times, trace)
    if show:
        do_show(times, {"trace": trace, "fit": a * np.sin(w * times + phi)})
    assert abs(w - 1.5) < 1e-2, f"Found w:{w}"
    assert abs(a - 9.9 * np.exp(-0.1 * 48.0)) < 1e-2, f"Amplitude {a} expected about {9.9 * np.exp(-0.1 * 48.0)}"
    logger.info(f"phi moved to {phi}. Basic phi was {np.pi / 4}")


def test_osc_fit(show: bool = False):
    """Check amplitude, frequency and phase of non-forced and forced oscillator in various configurations."""
    # non-forced oscillator. Various x0 and dx0 start values. No damping to make fit more exact.
    times = np.linspace(0.0, 50, 101)
    osc = ForcedOscillator1D(k=1.0, c=0.0, a=0.0, wf=0.5, d0=0.0)
    for x0 in np.linspace(0, 10, 11):
        for dx0 in np.linspace(-10, 10, 40):
            osc.coefficients(x0=x0, dx0=dx0)
            _x, _v = osc.calc(times)
            a, w, phi = sine_fit(times, _x)
            e_x0 = a * np.sin(phi)
            e_dx0 = w * a * np.cos(phi)
            if abs(x0 - e_x0) > 1e-10 or abs(dx0 - e_dx0) > 1e-10:
                if show:
                    do_show(
                        times,
                        {"x": _x, "v": _v, "e_x": a * np.sin(w * times + phi), "e_v": a * w * np.cos(w * times + phi)},
                        {"x": 1, "v": 2, "e_x": 1, "e_v": 2},
                    )
            assert abs(x0 - e_x0) < 1e-10, f"Non forced oscillator with x0={x0}, dx0={dx0}. curve-x0: {e_x0}"
            assert abs(dx0 - e_dx0) < 1e-10, f"Non forced oscillator with x0={x0}, dx0={dx0}. curve-dx0: {e_dx0}"

    # forced oscillator. Various x0, dx0, a, wf, d0 values. Damping to get to equilibrium.
    times = np.linspace(0.0, 200, 2001)  # longer time span to get to equilibrium
    osc = ForcedOscillator1D(k=1.0, c=0.2, a=1.0, wf=0.5, d0=0.0)
    for x0 in np.linspace(0, 10, 11):
        for dx0 in np.linspace(-2, 2, 20):
            for wf in np.linspace(0.8, 8.0, 9):
                for d0 in np.linspace(0.0, 7.0, 9):
                    osc.coefficients(x0=x0, dx0=dx0, wf=wf, d0=d0)
                    coef = f"x0:{osc.x0}, dx0:{osc.dx0}, A:{osc.A}, wf:{osc.wf}, d0:{d0}, d:{osc.d}"
                    _x, _v = osc.calc(times)
                    if abs(x0 - _x[0]) > 1e-10 or abs(dx0 - _v[0]) > 1e-10:  # check initial values
                        if show:
                            do_show(times, {"x": _x, "v": _v}, title=f"Forced x0={x0}, dx0={dx0}")
                        assert abs(x0 - _x[0]) < 1e-10, f"Forced oscillator with ({coef}). Curve-x0: {_x[0]}"
                        assert abs(dx0 - _v[0]) < 1e-10, f"Forced oscillator with ({coef}). Curve-dx0: {_v[0]}"

                    # check equilibrium values
                    a, w, phi = sine_fit(times, _x)
                    if abs(a - osc.A) > 1e-2 or abs(w - osc.wf) > 1e-2 or abs(phi - osc.d) > 1e-2:
                        if show:
                            do_show(
                                times,
                                {
                                    "x": _x,
                                    "v": _v,
                                    "e_x": a * np.sin(w * times + phi),
                                    "e_v": a * w * np.cos(w * times + phi),
                                },
                                {"x": 1, "v": 2, "e_x": 1, "e_v": 2},
                            )
                        assert abs(a - osc.A) < 1e-2, f"Forced oscillator with ({coef}). Equilibrium amplitude: {a}"
                        assert abs(w - osc.wf) < 1e-2, f"Forced oscillator with ({coef}). Equilibrium ang.freq: {w}"
                        assert abs(phi - osc.d) < 1e-2, f"Forced oscillator with ({coef}). Equilibrium phase: {phi}"


def test_oscillator(show: bool = False):
    """Test the Oscillator class in isolation.
    Such tests are strongly recommended before compiling the model into an FMU.

    With respect to `wiki <https://en.wikipedia.org/wiki/Oscillation>`_ our parameters are:
    b = c, k=k => beta = c/(2m), w0 = sqrt(k/m) => w1 = sqrt(beta^2 - w0^2) = sqrt( c^2/4/m^2 - k/m)
    """
    test_cases: list[tuple[float, float, float, float, float, str]] = [
        # k    c    m    a    w    x0   description
        (1.0, 0.0, 1.0, 0.0, 0.1, "Oscillator without damping and force"),
        (1.0, 0.2, 1.0, 0.0, 0.1, "Oscillator include damping"),
        (1.0, 2.0, 1.0, 0.0, 0.1, "Oscillator critically damped"),
        (1.0, 5.0, 1.0, 0.0, 0.1, "Oscillator over-damped"),
        (1.0, 0.2, 1.0, 1.0, 0.5, "Forced oscillation. Less than resonance freq"),
        (1.0, 0.2, 1.0, 1.0, 1.0, "Forced oscillation. Damped. Resonant"),
        (1.0, 0.2, 1.0, 1.0, 2.0, "Forced oscillation. Damped. Above resonance freq"),
        (1.0, 2.0, 1.0, 1.0, 1.0, "Forced oscillation. Crit. damped. Resonant"),
        (1.0, 5.0, 1.0, 1.0, 1.0, "Forced oscillation. Over-damped. Resonant"),
        (1.0, 2.0, 1.0, 1.0, 2.0, "Forced oscillation. Crit. damped. Above resonance freq"),
        (1.0, 5.0, 1.0, 1.0, 2.0, "Forced oscillation. Over-damped. Above resonance freq"),
    ]
    for k, c, m, a, w, msg in test_cases:
        logger.info(f"{msg}. k:{k}, c:{c}, a:{a}, w:{w}")
        osc = ForcedOscillator1D(k, c, m, a, w)
        logger.info(f"Coefficients {msg}: c1={osc.c1}, c2={osc.c2}, phi={osc.phi}")
        if c < 2.0:  # only if damping is small enough
            cp = 2.0 * np.pi / np.sqrt(k / m - (c / 2.0 / m) ** 2)
            assert abs(osc.period() - cp) < 1e-12, f"Period: {osc.period} != {2 * np.pi}"
        times = np.linspace(0.0, 20, 10001)
        _x, _v = osc.calc(times)

        # Check of velocity, based on numerical calculation of derivative of position
        dt = times[1] - times[0]
        for i in range(2, len(times) - 2):
            assert abs((-_x[i + 2] + 8 * _x[i + 1] - 8 * _x[i - 1] + _x[i - 2]) / (12 * dt) - _v[i]) < 1e-1, (
                f"{msg}: Derivative@{times[i]} {_v[i]} != ({-_x[i + 2]}+8*{_x[i + 1]}-8*{_x[i - 1]}+{_x[i - 2]}) / {12 * dt}"
            )

        # check of start position and velocity (0.0, 1.0) for non-forced and (0.0, 0.0) for forced
        assert abs(_x[0]) < 1e-15, f"'{msg}': Start position {_x[0]} != 0.0"
        if a == 0:  # not forced. x0=0, dx0=1.0
            assert abs(_v[0] - 1.0) < 1e-15, f"'{msg}': Start velocity {_v[0]} != 1.0"
        else:  # forced. x0=dx0=0.0
            assert abs(_v[0]) < 1e-14, f"Start velocity {_v[0]} != 0.0 in {msg}"
        if show:
            do_show(times, {"x": _x, "v": _v}, {"x": 1, "v": 2}, title=msg)


def test_single(show: bool = False):
    """Test single cases which caused problems."""
    times = np.linspace(0.0, 100, 2001)  # longer time span to get to equilibrium
    osc = ForcedOscillator1D(k=1.0, c=0.2, a=1.0, x0=0.0, dx0=0.0, wf=0.5)
    coef = f"x0:{osc.x0}, dx0:{osc.dx0}, A:{osc.A}, wf:{osc.wf}, d0:{osc.d0}, d:{osc.d}"
    logger.info(f"Oscillator with {coef}")
    _x, _v = osc.calc(times)
    a, w, phi = sine_fit(times, _x)
    e_x = a * np.sin(w * times + phi)
    e_v = a * w * np.cos(w * times + phi)
    if show:
        do_show(times, {"x": _x, "v": _v, "e_x": e_x, "e_v": e_v}, {"x": 1, "v": 2, "e_x": 1, "e_v": 2})
    assert abs(phi - osc.d) < 1e-2, f"Forced oscillator with ({coef}). Equilibrium phase: {phi}"


if __name__ == "__main__":
    retcode = pytest.main(["-rA", "-v", "--rootdir", "../", "--show", "False", __file__])
    assert retcode == 0, f"Non-zero return code {retcode}"

    # test_amplitude_omega_phase(show=True)
    # test_osc_fit(show=True)
    # test_oscillator(show=True)
    # test_single(show=True)
