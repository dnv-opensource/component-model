import logging

import matplotlib.pyplot as plt
import numpy as np
import pytest

from component_model.utils.analysis import extremum, extremum_series, sine_fit

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def test_extremum():
    t = [np.radians(10 * x) for x in range(100)]
    x = [np.cos(x) for x in t]
    e, p = extremum(t[0:3], x[0:3], 2e-3)  # allow a small error
    assert e == 1
    assert p[0] > -2e-3 and p[1] < 1 + 1e-6, (
        "Top of parabola somewhat to the left due to cos not exactly equal to 2.order"
    )
    # for i in range(100):
    #    print(i, t[i], x[i])
    e, p = extremum(t[17:20], x[17:20])
    assert e == -1 and abs(p[0] - np.pi) < 1e-10 and np.isclose(p[1], -1)
    ex = extremum_series(t, x, "all")
    assert len(ex) == 2
    assert np.allclose(ex[0], (12.566370614359142, 1.0))
    assert np.allclose(ex[1], (15.707963267948958, -1.0))


def test_sine_fit(show: bool = False):
    def do_test(y: np.ndarray, idx: int = 1):
        y0, a, w, phi, tm = sine_fit(times, y)
        if show:
            fig, ax = plt.subplots()
            ax.plot(times, y)
            ax.plot(times, y0 + a * np.sin(w * times + phi))
            plt.show()
        _y0 = (0, 4.4, 0)[idx]
        _w = (2.2, 2.2, 2 * np.pi / 9.0)[idx]
        _phi = (1.1, 1.1, np.pi / 2)[idx]
        assert abs(y0 - _y0) < 1e-6, f"[{idx}]. Translation {y0} != {_y0}"
        assert abs(a - 3.3) < 1e-6, f"[{idx}]. Amplitude {a} != 3.3"
        assert abs(w - _w) < 1e-6, f"[{idx}]. Angular freq {w} != {_w}"
        assert abs(phi - _phi) < 1e-6, f"[{idx}]. Phase {phi} != {_phi}"

    times = np.linspace(0, 10, 100)
    do_test(3.3 * np.sin(2.2 * times + 1.1), idx=0)
    do_test(4.4 + 3.3 * np.sin(2.2 * times + 1.1), idx=1)
    do_test(3.3 * np.cos(2 * np.pi / 9.0 * times), idx=2)


if __name__ == "__main__":
    retcode = pytest.main(["-rP -s -v", "--show", "False", __file__])
    assert retcode == 0, f"Return code {retcode}"
    # test_extremum()
    # test_sine_fit(show=True)
