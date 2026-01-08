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
        a,wf (float): sinusoidal force parameters (amplitude and angular frequency)
        x0, v0 (float): start values for harmonic oscillator (force has fixed start values)
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
            assert x0 is None and v0 is None, "Checking of forced oscillations is only implemented for x0, v0=None"
            self.A = a / np.sqrt(((self.w0**2 - wf**2) ** 2 + (2 * self.b * wf) ** 2))  # equilibrium amplitude
            self.d = np.arctan2(2 * self.b * wf, self.w0**2 - wf**2)  # phase angle in equilibrium
            self.x0 = self.A * np.sin(self.d)
            self.v0 = -self.A * wf * np.cos(self.d)
        else:
            self.x0 = x0
            self.v0 = v0 if v0 is not None else 0.0

    def calc(self, t: float | np.ndarray):
        a, d, b, wd, x0, v0, wf = self.a, self.d, self.b, self.wd, self.x0, self.v0, self.wf
        assert x0 is not None
        assert v0 is not None
        if a != 0.0:
            x_e = self.A * np.sin(wf * t - d)
            v_e = self.A * wf * np.cos(wf * t - d)
        else:
            x_e, v_e = (0.0, 0.0)

        if not np.isnan(wd) and wd > 1e-10:  # damped oscillation
            x = np.exp(-b * t) * (x0 * np.cos(wd * t) + (x0 * b + v0) / wd * np.sin(wd * t))
            v = np.exp(-b * t) * (v0 * np.cos(wd * t) - (wd * x0 + b**2 / wd * x0 + b / wd * v0) * np.sin(wd * t))
        elif not np.isnan(wd) and wd < 1e-10:  # critically damped oscillation
            x = ((v0 + b * x0) * t + x0) * np.exp(-b * t)
            v = -b * (v0 + b * x0) * t * np.exp(-b * t)
        else:  # over-damped oscillation
            _b1 = b + np.sqrt(b**2 - self.w0**2)
            _b2 = b - np.sqrt(b**2 - self.w0**2)

            x = ((_b2 * x0 + v0) * np.exp(-_b1 * t) - (_b1 * x0 + v0) * np.exp(-_b2 * t)) / (_b2 - _b1)
            v = -(_b1 * (_b2 * x0 + v0) * np.exp(-_b1 * t) - _b2 * (_b1 * x0 + v0) * np.exp(-_b2 * t)) / (_b2 - _b1)

        if a != 0:
            return (x + x_e, v + v_e)
        else:
            return (x, v)
