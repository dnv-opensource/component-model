import logging

import numpy as np

logger = logging.getLogger(__name__)


def extremum(
    x: tuple[float, ...] | list[float] | np.ndarray, y: tuple[float, ...] | list[float] | np.ndarray, aerr: float = 0.0
):
    """Check whether the provided (3) points contain an extremum.
    Return 0 (no extremum), -1 (low point), 1 (top point) and the point, or (0,0).
    """
    assert len(x) == 3 and len(y) == 3, f"Exactly three points expected. Found {x}, {y}"
    a = np.array(((1, x[0], x[0] ** 2), (1, x[1], x[1] ** 2), (1, x[2], x[2] ** 2)), float)
    z = np.linalg.solve(a, y)
    if (abs(z[1]) < 1e-15 and abs(z[2]) < 1e-15) or abs(z[2]) < abs(z[1] * 1e-15):  # very nearly linear.
        return (0, (0, 0))
    else:
        x0 = -z[1] / 2.0 / z[2]
        if x[0] - aerr <= x0 <= x[2] + aerr:  # extremum in the range
            z0 = z[0] + (z[1] + z[2] * x0) * x0
            if z[2] < 0:
                return (1, (x0, z0))
            else:
                return (-1, (x0, z0))
        else:
            return (0.0, (0.0, 0.0))


def extremum_series(
    t: tuple[float, ...] | list[float] | np.ndarray, y: tuple[float, ...] | list[float] | np.ndarray, which: str = "max"
):
    """Estimate the extrema from the time series defined by y(t).
    which can be 'max', 'min' or 'all'.
    """

    def w1(x: float):
        return x

    def w_1(x: float):
        return -x

    def w0(x: float):
        return abs(x)

    assert len(t) == len(y) > 2, "Something wrong with lengths of t ({len(t)}) and y ([len(y)})"
    if which == "max":
        w = w1
    elif which == "min":
        w = w_1
    else:
        w = w0

    res = []
    for i in range(1, len(t) - 1):
        e, p = extremum(t[i - 1 : i + 2], y[i - 1 : i + 2])
        if e != 0 and w(e) == 1 and p[0] < t[i]:
            res.append(p)
    return res
