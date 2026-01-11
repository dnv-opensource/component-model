"""Collect analytic models in this module, as facility to perform simple verifications on simulation models."""

import numpy as np


class ForcedOscillator1D(object):
    """Calculate the expected (analytic) position and speed of a harmonic oscillator (in one dimension)
    with the given parameter setting.

    i.e. the solution to the differential equation:

    m* d2x + c* dx + k* x = a* sin(wf*t) , with x(0)=x0 and dx(0)=v0

    Used for verification purposes, but made available within code base to be available within other packages.

    Args:
        k,c,m (float): harmonic oscillator parameters
        a,wf (float): sinusoidal force parameters (amplitude and angular frequency). Assume F(t) = a*sin(wf*t)
        x0, v0 (float): start values for harmonic oscillator (force has fixed start values).
           Assume x0=v0=0.0 for forced oscillation and x0=0, v0=1 for non-forced if None.
    """

    def __init__(
        self,
        k: float,
        c: float,
        m: float = 1.0,
        a: float = 0.0,
        wf: float = 0.1,
        x0: float | None = None,
        v0: float | None = None,
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
        self.wf = wf
        self.d = 0.0
        if a != 0:
            self.A = a / np.sqrt(((self.w0**2 - wf**2) ** 2 + (2 * self.b * wf) ** 2))  # equilibrium amplitude
            self.d = np.arctan2(2 * self.b * wf, self.w0**2 - wf**2)  # phase angle in equilibrium
        else:
            self.A = 0.0
            self.d = 0.0
        self.coefficients(x0, v0)  # coefficients. Sets and returns self.x0, self.v0, self.c1, self.c2, self.phi

    def calc(self, t: float | np.ndarray, wf: float | None = None, x0: float | None = None, v0: float | None = None):
        a, d, b, wd = self.a, self.d, self.b, self.wd
        if wf is None:
            wf = self.wf
        if x0 is None:
            x0 = self.x0
        if v0 is None:
            v0 = self.v0
        assert x0 is not None
        assert v0 is not None

        if not np.isnan(wd) and wd > 1e-10:  # damped oscillation
            assert self.c1 is not None and self.phi is not None, "c1 and phi needed for damped oscillation"
            x = self.c1 * np.exp(-b * t) * np.sin(wd * t + self.phi)
            v = self.c1 * np.exp(-b * t) * (self.wd * np.cos(wd * t + self.phi) - self.b * np.sin(wd * t + self.phi))
        elif not np.isnan(wd) and wd < 1e-10:  # critically damped oscillation
            assert self.c1 is not None and self.c2 is not None, "c1 and c2 needed for critically damped oscillation"
            x = (self.c1 + self.c2 * t) * np.exp(-b * t)
            v = (self.c2 - b * (self.c1 + self.c2 * t)) * np.exp(-b * t)
        else:  # over-damped oscillation
            assert self.c1 is not None and self.c2 is not None, "c1 and c2 needed for over-damped oscillation"            
            _sqrt = np.sqrt(self.b**2 - self.w0**2)
            x = self.c1 * np.exp((-b + _sqrt) * t) + self.c2 * np.exp((-b - _sqrt) * t)
            v = self.c1 * (-b + _sqrt) * np.exp((-b + _sqrt) * t) + self.c2 * (-b - _sqrt) * np.exp((-b - _sqrt) * t)

        if a != 0.0:
            x += self.A * np.sin(wf * t - d)
            v += self.A * wf * np.cos(wf * t - d)

        return (x, v)

    def coefficients(
        self, x0: float | None = None, dx0: float | None = None, y0: float | None = None, dy0: float | None = None
    ):
        """Calculate integration parameters (C1 and C2/phi)
        such that start position x0 and velocity dx0 are as specified.
        Pre-calculated parameters for force y0 and dy0 can be added.

        Default values x0, dx0: (0.0, 1.0) or (1.0, 0.0) if not forced. (0.0, 0.0) if forced.
        Default values y0, dy0: (0.0, 0.0) if not forced. (A*sin(d), wf*A*cos(d)) if forced.
        """
        if self.a == 0.0:  # not forced
            if x0 is None and dx0 is None:
                x0, dx0 = (0.0, 1.0)
            elif dx0 is not None:
                x0, dx0 = (0.0 if dx0 > 0 else 1.0, dx0)
            elif x0 is not None:
                x0, dx0 = (x0, 0.0 if x0 > 0 else 1.0)
            if y0 is None:
                y0 = 0.0
            if dy0 is None:
                dy0 = 0.0
        else:  # forced
            x0 = 0.0 if x0 is None else x0
            dx0 = 0.0 if dx0 is None else dx0
            if y0 is None:
                y0 = -self.A * np.sin(self.d)
            if dy0 is None:
                dy0 = self.wf * self.A * np.cos(self.d)

        assert x0 is not None and y0 is not None, "x0 and y0 must be defined at this stage"
        assert dx0 is not None and dy0 is not None, "dx0 and dy0 must be defined at this stage"
        if self.wd > 0.0:  # damped oscillation
            if abs(x0 - y0) < 1e-6:
                self.c1 = (self.b * (x0 - y0) + (dx0 - dy0)) / self.wd
                self.phi = (x0 - y0) / self.c1
            else:
                self.phi = np.arctan2(self.wd, self.b + (dx0 - dy0) / (x0 - y0))
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
        self.x0 = x0
        self.v0 = dx0
        return (x0, dx0, self.c1, self.c2, self.phi)

    def period(self):
        if self.wd > 0:
            return 2 * np.pi / self.wd
        else:
            return float("inf")
