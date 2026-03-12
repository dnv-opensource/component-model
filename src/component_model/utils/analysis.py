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

    assert len(t) == len(y), f"Length of sequences t:{len(t)} != y:{len(y)}"
    assert len(t) > 2, f"Sequences t and y shall be longer than 2, but length was {len(t)} for both"
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


def sine_fit(times: list[float] | np.ndarray, vals: list[float] | np.ndarray, eps: float = 1e-2):
    """Fit a general sine function f(t) = y0 + a* sin(w*t + phi) to the data end and return (y0, a,omega,phi).

    * The last two maximum points of the data set are detected and a full sine wave is fit to that.
    * Error is issued if maximum points are not found.
    * Warning is provided if the fit to the sine function is bad (diag(pcov) > eps).
    * If the curve starts with maxima (cos(...)), it is accepted if the first points fit a 2nd order within eps.
    * The phase angle is returned in the range ]-pi, pi]
    * Returns the zero-line, the amplitude, the angualar frequency, the phase and the mid-time of the wave.
    """
    from scipy.optimize import curve_fit

    def do_fit(x: np.ndarray, y: np.ndarray):
        """Perform a sin function fit on the given data and return parameters."""

        def func(x: float | np.ndarray, y0: float, a: float, w: float, phi: float, /):
            return y0 + a * np.sin(w * x + phi)

        avg = np.average(y)
        popt, pcov = curve_fit(func, x, y, p0=(avg, y[0] - avg, 2 * np.pi / (x[-1] - x[0]), np.pi / 2))
        if max(np.diag(pcov)) > eps:
            logger.warning(f"Curve cannot be fitted well to sine function: {np.diag(pcov)}. Detected t0:{x[0]}")
        return popt

    def np_index(arr: np.ndarray, t: float):
        """Get the closest index with respect to a value in the array."""
        return np.absolute(arr - t).argmin()

    times = np.array(times, float)
    vals = np.array(vals)
    state = 1
    top1, top0 = None, None
    for t1, v1, t2, v2 in reversed(list(zip(times[:-1], vals[:-1], times[1:], vals[1:], strict=True))):
        if state == 1:  # upward slope at end of curve
            if v2 <= v1:
                state = 2
        elif state == 2:  # decreasing curve. Search for last top point
            if v2 > v1:  # found the last top point
                top1 = np_index(times, t2)
                state = 3
        elif state == 3:  # increasing curve
            if v2 <= v1:
                state = 4
        elif state == 4:
            if v2 > v1:  # found the last top point
                top0 = np_index(times, t1)
                state = 5
            elif t1 == times[0]:  # first point and first maximum not yet identified. Check whether max(2nd order)
                t3 = times[2]
                v3 = vals[2]
                c = ((t3 - t1) * (v2 - v1) - (t2 - t1) * (v3 - v1)) / (
                    (t3 - t1) * (t2**2 - t1**2) - (t2 - t1) * (t3**2 - t1**2)
                )
                b = ((v2 - v1) - c * (t2**2 - t1**2)) / (t2 - t1)
                a = v1 - b * t1 - c * t1**2
                if abs(b + 2 * c * t1) < eps and c < 0:  # accept as start of curve
                    top0 = 0
                    state = 5
        elif state == 5:
            break
    if state == 4:
        raise AssertionError(f"Only one maximum {vals[top1]}@{times[top1]} found.") from None
    elif state < 4:
        raise AssertionError("Maxima not found") from None
    t0 = times[top0]

    y0, a, w, phi = do_fit(times[top0:top1] - t0, vals[top0:top1])
    phi -= w * t0  # correct for the time translation
    if w < 0:
        a, w, phi = -a, -w, -phi
    if a < 0:
        a = -a
        phi += np.pi
    # phi -= w * t0 % (2 * np.pi)
    while phi > np.pi - 1e-14:
        phi -= 2 * np.pi
    while phi <= -np.pi + 1e-14:
        phi += 2 * np.pi
    assert top0 is not None and top1 is not None
    return (y0, a, w, phi, times[int((top0 + top1) / 2)])
