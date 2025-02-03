from collections.abc import Callable

import numpy as np
from scipy import integrate


class HarmonicOscillator:
    """Construct a simple model of a general harmonic oscillator, potentially driven by a force.

    The system obeys the equation F(t) - k*x - c*dx/dt = m*d^2x/dt^2
    See also `Wikipedia <https://en.wikipedia.org/wiki/Harmonic_oscillator>`_

    where x shall be a 3D vector with an initial position. F(t)=0 as long as there is not external driving force.
    NOTE: This is the basic oscillator model.

    * Unrelated with Model and Variable
    * FMU cannot be built from that! See oscillator_fmu where this is extended to an FMU package.
    * The HarmonicOscillator can be used for model testing (highly recomended, see test_oscillator.py)
    * It can of cause be used stand-alone

    We use the scipy.integrate.solve_ivp algorithm for the integration (see do_step)

    Args:
        k (tuple)=(1.0, 1.0, 1.0): spring constant in N/m. May vary in 3D
        c (tuple)=(0.0, 0.0, 0.0): Viscous damping coefficient in N.s/m. May vary in 3D
        m (float)=1.0: Mass of the spring load (spring mass negligible) in kg
        tolerance (float)=1e-5: Optional tolerance in m, i.e. maximum uncertainty in displacement x.
    """

    def __init__(
        self,
        k: tuple[float, float, float] | tuple[str, str, str] = (1.0, 1.0, 1.0),  # type str for FMU option
        c: tuple[float, float, float] | tuple[str, str, str] = (0.0, 0.0, 0.0),
        m: float = 1.0,
        tolerance: float = 1e-5,
        **kwargs,
    ):
        self.k = np.array(k, float)
        self.c = np.array(c, float)
        self.m = m
        self.tolerance = tolerance
        self.x = np.array((0, 0, 0), float)
        self.v = np.array((0, 0, 0), float)
        self.f = np.array((0, 0, 0), float)
        self.fi = 0
        # standard ODE matrix (homogeneous system):
        self.ode = [np.array(((-self.c[i] / self.m, -self.k[i] / self.m), (1, 0)), float) for i in range(3)]

    def ode_func(self, t: float, y: np.ndarray, i: int, f: float) -> np.ndarray:
        res = self.ode[i].dot(y)
        if f != 0:
            res += np.array((f, 0), float)
        return res

    def do_step(self, time: int | float, dt: int | float) -> bool:
        """Do one simulation step of size dt.

        We implement a very simplistic algoritm based on difference calculus.
        """
        for i in range(3):  # this is a 3D oscillator
            if self.x[i] != 0 or self.v[i] != 0 or self.f[i] != 0:
                y0 = np.array([self.v[i], self.x[i]], float)
                sol = integrate.solve_ivp(
                    fun=self.ode_func,
                    t_span=[time, time + dt],
                    y0=y0,
                    args=(i, self.f[i]),  # dimension and force as extra arguments to fun
                    atol=self.tolerance,
                )  # , method='DOP853') #RK45 100x less accurate
                self.x[i] = sol.y[1][-1]
                self.v[i] = sol.y[0][-1]
        return True  # to keep the signature when moving to FMU

    @property
    def period(self):
        """Calculate the natural period of the oscillator (without damping an)."""
        w2 = []
        for i in range(3):
            w = self.k[i] / self.m - (self.c[i] / 2 / self.m) ** 2
            if w > 0:
                w2.append(2 * np.pi / np.sqrt(w))
            else:
                w2.append(float("nan"))  # critically or over-damped. There is no period
        return w2


class DrivingForce:
    """A driving force in 3 dimensions which produces an ouput per time and can be connected to the oscillator.

    Args:
        func (callable)=lambda t:np.array( (0,0,0), float): A function of t, producing a 3D vector
    """

    def __init__(self, func: Callable):
        self.func = func
        self.out = np.array((0, 0, 0), float)

    def do_step(self, time: float, dt: float):
        self.out = self.func(time)
