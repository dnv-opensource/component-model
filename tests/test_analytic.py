import logging

import matplotlib.pyplot as plt
import numpy as np
import pytest

from component_model.analytic import ForcedOscillator1D

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def do_show(
    times: list[float] | np.ndarray,
    traces: dict[str, list[list[float]]],
    selection: dict[str, int] | None = None,
    title: str = ""
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
        print(f"Coefficients {msg}: c1={osc.c1}, c2={osc.c2}, phi={osc.phi}")
        if c < 2.0:  # only if damping is small enough
            cp = 2.0 * np.pi / np.sqrt(k / m - (c / 2.0 / m) ** 2)
            assert abs(osc.period() - cp) < 1e-12, f"Period: {osc.period} != {2 * np.pi}"
        times = np.linspace(0.0, 20, 10001)
        _x, _v = osc.calc(times)
        dt = times[1] - times[0]
        for i in range(2, len(times) - 2):
            assert abs((-_x[i + 2] + 8 * _x[i + 1] - 8 * _x[i - 1] + _x[i - 2]) / (12 * dt) - _v[i]) < 1e-1, (
                f"{msg}: Derivative@{times[i]} {_v[i]} != ({-_x[i + 2]}+8*{_x[i + 1]}-8*{_x[i - 1]}+{_x[i - 2]}) / {12 * dt}"
            )
        assert abs(_x[0]) < 1e-14, f"'{msg}': Start position {_x[0]} != 0.0"
        if a == 0:  # not forced. x0=0, v0=1.0
            assert abs(_v[0] - 1.0) < 1e-14, f"'{msg}': Start velocity {_v[0]} != 1.0"
        else:  # forced. x0=v0=0.0
            #            assert abs(_v[0]) < 1e-14, f"Start velocity {_v[0]} != 0.0 in {msg}"
            print(f"Start velocity {_v[0]} != 0.0 in {msg}")
        if show:
            do_show(times, {"x": _x, "v": _v}, {"x": 1, "v": 2}, title=msg)


if __name__ == "__main__":
    retcode = pytest.main(["-rA", "-v", "--rootdir", "../", "--show", "False", __file__])
    assert retcode == 0, f"Non-zero return code {retcode}"

    # test_oscillator(show=True)
    # test_special(show=True)
