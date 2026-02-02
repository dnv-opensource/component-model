"""Collect analytic models in this module, as facility to perform simple verifications on simulation models."""

import logging

import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ForcedOscillator1D(object):
    """Calculate the expected (analytic) position and speed of a harmonic oscillator in one dimension
    with the given parameter settings.

    i.e. the solution to the differential equation:

    m* d2x + c* dx + k* x = a* sin(wf*t+d0) , with x(0)=x0 and dx(0)=dx0

    Used for verification purposes, but made available within code base to be available for other packages.

    Args:
        k,c,m (float): harmonic oscillator parameters
        a,wf (float): sinusoidal force parameters (amplitude and angular frequency). Assume F(t) = a*sin(wf*t)
        x0, dx0 (float): start values for harmonic oscillator (force has fixed start values).
           Assume x0=dx0=0.0 for forced oscillation and x0=0, dx0=1 for non-forced if None.
        d0 (float): optional additional phase of force. E.g. d0=np.pi/2 leads to a cos(wf*t) force
    """

    def __init__(
        self,
        k: float,
        c: float,
        m: float = 1.0,
        a: float = 0.0,
        wf: float = 0.1,
        x0: float | None = None,
        dx0: float | None = None,
        d0: float = 0.0,
    ):
        self.w0 = np.sqrt(k / m)  # omega0
        self.b = c / (2 * m)  # beta
        if self.w0 > self.b:
            self.wd = np.sqrt(self.w0**2 - self.b**2)  # angular frequency of oscillation
        elif abs(self.w0 - self.b) < 1e-13:  # aperiodic limiting case
            self.wd = 0.0
        else:
            self.wd = np.nan
        self.a = a
        self.x0 = 0.0 if x0 is None else x0
        if a == 0.0:  # non-forced
            self.dx0 = 1.0 if dx0 is None else dx0
        else:  # forced
            self.dx0 = 0.0 if dx0 is None else dx0
        self.wf = wf
        self.d: float
        self.d0 = d0
        self.y0: float
        self.dy0: float
        self.c1: float
        self.c2: float | None
        self.phi: float | None
        # equilibrium amplitude and phase of (forced) oscillator
        self.coefficients(x0, dx0, a, wf, d0)  # Sets and returns .c1, .c2, .phi, .A, .d + .x0, .dx0, .y0, .dy0

    def calc(self, t: float | np.ndarray):
        """Calculate position x and speed v of oscillator for time(s) t.
        Change of any of .x0, .dx0, .a, .wf, .d0 requires running  .coefficients() beforehand.
        """
        a, d, b, w0, wd = self.a, self.d, self.b, self.w0, self.wd
        A, wf, d = self.A, self.wf, self.d
        c1, c2, phi = self.c1, self.c2, self.phi

        if not np.isnan(wd) and wd > 1e-10:  # damped oscillation
            assert c1 is not None and phi is not None, "c1 and phi needed for damped oscillation"
            x = c1 * np.exp(-b * t) * np.sin(wd * t + phi)
            v = c1 * np.exp(-b * t) * (wd * np.cos(wd * t + phi) - b * np.sin(wd * t + phi))
        elif not np.isnan(wd) and wd < 1e-10:  # critically damped oscillation
            assert c1 is not None and c2 is not None, "c1 and c2 needed for critically damped oscillation"
            x = (c1 + c2 * t) * np.exp(-b * t)
            v = (c2 - b * (c1 + c2 * t)) * np.exp(-b * t)
        else:  # over-damped oscillation
            assert c1 is not None and c2 is not None, "c1 and c2 needed for over-damped oscillation"
            _sqrt = np.sqrt(b**2 - w0**2)
            x = c1 * np.exp((-b + _sqrt) * t) + c2 * np.exp((-b - _sqrt) * t)
            v = c1 * (-b + _sqrt) * np.exp((-b + _sqrt) * t) + c2 * (-b - _sqrt) * np.exp((-b - _sqrt) * t)

        if a != 0.0:  # forced oscillation
            x += A * np.sin(wf * t + d)
            v += A * wf * np.cos(wf * t + d)

        return (x, v)

    def coefficients(
        self,
        x0: float | None = None,
        dx0: float | None = None,
        a: float | None = None,
        wf: float | None = None,
        d0: float | None = None,
    ):
        """Calculate integration parameters (C1 and C2/phi, and A, d)
        such that start position x0 and velocity dx0 and a, wf and d0 are as specified.

        Default values x0, dx0: (0.0, 1.0) or (1.0, 0.0) if not forced. (0.0, 0.0) if forced.
        Default values a, wf, d0: the stored value self.*
        Note: d0=0.0 results into a sine function as force while e.g. d0=pi/2 results into a cosine force.
        """
        if a is not None or wf is not None or d0 is not None:  # something changed within force
            if a is not None:
                self.a = a
            if wf is not None:
                self.wf = wf
            if d0 is not None:
                self.d0 = d0

            if self.a == 0.0:  # non-forced
                self.y0, self.dy0 = 0.0, 0.0
                self.A, self.d = 0.0, 0.0
            else:  # forced
                self.A = self.a / np.sqrt(((self.w0**2 - self.wf**2) ** 2 + (2 * self.b * self.wf) ** 2))
                self.d = np.arctan2(2 * self.b * self.wf, self.w0**2 - self.wf**2) - self.d0
                while self.d < -np.pi:
                    self.d += 2 * np.pi
                self.y0 = self.A * np.sin(self.d)
                self.dy0 = self.wf * self.A * np.cos(self.d)
        y0, dy0 = self.y0, self.dy0

        # changes within x0 or dx0
        if x0 is None:
            x0 = self.x0
        else:
            self.x0 = x0
        if dx0 is None:
            dx0 = self.dx0
        else:
            self.dx0 = dx0

        # calculate c1, c2/phi
        assert x0 is not None and y0 is not None, "x0 and y0 must be defined at this stage"
        assert dx0 is not None and dy0 is not None, "dx0 and dy0 must be defined at this stage"
        if self.wd > 0.0:  # damped oscillation
            if abs(x0 - y0) < 1e-6:  # special case which would lead to division by zero
                self.c1 = (self.b * (x0 - y0) + (dx0 - dy0)) / self.wd
                self.phi = (x0 - y0) / self.c1
            else:
                self.phi = np.arctan2(self.wd, self.b + (dx0 - dy0) / (x0 - y0))
                assert self.phi is not None
                self.c1 = (x0 - y0) / np.sin(self.phi)
            self.c2 = None
        elif self.wd == 0:  # critically damped oscillation
            self.c1 = x0 - y0
            self.c2 = self.b * (x0 - y0) + dx0 - dy0
            self.phi = None
        else:  # over-damped oscillation
            _sqrt = np.sqrt(self.b**2 - self.w0**2)
            self.c1 = -((-self.b - _sqrt) * (x0 - y0) - (dx0 - dy0)) / 2 / _sqrt
            self.c2 = ((-self.b + _sqrt) * (x0 - y0) - (dx0 - dy0)) / 2 / _sqrt
            self.phi = None
        return (self.c1, self.c2, self.phi, self.A, self.d)

    def period(self):
        if self.wd > 0:
            return 2 * np.pi / self.wd
        else:
            return float("inf")


def sine_fit(times: list[float] | np.ndarray, vals: list[float] | np.ndarray, max_pcov: float = 1e-2):
    """Fit a general sine function f(t) = a* sin(w*t + phi) to the data and return (a,omega,phi).

    * The last two upward zero-crossings of the data set are detected and a full sine wave is fit to that.
    * Error is issued if zero-crossings are not found.
    * Warning is provided if the fit to the sine function is bad (diag(pcov) > max_pcov).
    * The phase angle is returned in the range ]-pi, pi]
    """
    from scipy.optimize import curve_fit

    def do_fit(x: np.ndarray, y: np.ndarray):
        """Perform a sind function fit on the given data and return parameters."""

        def func(x: float | np.ndarray, a: float, w: float, phi: float, /):
            return a * np.sin(w * x + phi)

        popt, pcov = curve_fit(func, x, y)
        if max(np.diag(pcov)) > max_pcov:
            logger.warning(f"Curve cannot be fitted well to sine function: {np.diag(pcov)}. Detected to:{x[0]}")
        return popt

    times = np.array(times, float)
    vals = np.array(vals)
    zero0, zero1, previous = None, None, None
    for t, v in reversed(list(zip(times, vals, strict=True))):
        if previous is not None:
            if np.sign(v) == 0 and np.sign(previous[1]) == 1:
                if zero1 is None:
                    zero1 = list(times).index(t)
                else:
                    zero0 = list(times).index(t)
                    break
            elif np.sign(v) == -1 and np.sign(previous[1]) == 1:
                if zero1 is None:
                    zero1 = list(times).index(previous[0])
                else:
                    zero0 = list(times).index(t)
                    break
        previous = (t, v)
    assert zero0 is not None and zero1 is not None, "Zeroes not found"
    t0 = times[zero0]

    a, w, phi = do_fit(times[zero0 : zero1 + 1] - t0, vals[zero0 : zero1 + 1])
    if w < 0:
        a, w, phi = -a, -w, -phi
    if a < 0:
        a = -a
        phi += np.pi
    phi -= w * t0 % (2 * np.pi)
    while phi > np.pi - 1e-14:
        phi -= 2 * np.pi
    while phi <= -np.pi + 1e-14:
        phi += 2 * np.pi
    return a, w, phi
