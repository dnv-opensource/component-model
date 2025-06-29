from typing import Callable

import numpy as np
from scipy import integrate


class Oscillator:
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
        m (float | tuple)=1.0: Mass of the spring load (spring mass negligible) in kg. Same in all dim. if float
        tolerance (float)=1e-5: Optional tolerance in m, i.e. maximum uncertainty in displacement x.
    """

    def __init__(
        self,
        k: tuple[float, ...] | tuple[str, ...] = (1.0, 1.0, 1.0),  # type str for FMU option
        c: tuple[float, ...] | tuple[str, ...] = (0.0, 0.0, 0.0),
        m: float | tuple[float, ...] = 1.0,
        tolerance: float = 1e-5,
        f_func: Callable | None = None,
    ):
        self.dim = len(k)
        self.k = np.array(k, float)
        self.c = np.array(c, float)
        if isinstance(m, float):
            self.m = np.array((m,) * self.dim, float)
        else:
            self.m = np.array(m, float)
        self.tolerance = tolerance
        self.x = np.array((0,) * self.dim, float)
        self.v = np.array((0,) * self.dim, float)
        self.f = np.array((0,) * self.dim, float)
        # standard ODE matrix (homogeneous system):
        self.ode = [
            np.array(((-self.c[i] / self.m[i], -self.k[i] / self.m[i]), (1, 0)), float) for i in range(self.dim)
        ]
        self.f_func = f_func

    def ode_func(self, t: float, y: np.ndarray, i: int, f: float) -> np.ndarray:
        res = self.ode[i].dot(y)
        if self.f_func is None:
            if f != 0:
                res += np.array((f, 0), float)
        elif i == 2:  # only implemented for z
            res += np.array((self.f_func(t)[i], 0), float)
        return res

    def do_step(self, current_time: float, step_size: int | float) -> bool:
        """Do one simulation step of size dt.

        We implement a very simplistic algoritm based on difference calculus.
        """
        print(f"OSC do_step")
        for i in range(self.dim):  # this is a xD oscillator
            if self.x[i] != 0 or self.v[i] != 0 or self.f[i] != 0 or self.f_func is not None:
                y0 = np.array([self.v[i], self.x[i]], float)
                sol = integrate.solve_ivp(
                    fun=self.ode_func,
                    t_span=[current_time, current_time + step_size],
                    y0=y0,
                    args=(i, self.f[i]),  # axis and force as extra arguments to fun
                    atol=self.tolerance,
                )
                self.x[i] = sol.y[1][-1]
                self.v[i] = sol.y[0][-1]
        return True  # to keep the signature when moving to FMU

    @property
    def period(self):
        """Calculate the natural period of the oscillator (without damping an)."""
        w2 = []
        for i in range(self.dim):
            w2i = self.k[i] / self.m[i] - (self.c[i] / 2 / self.m[i]) ** 2
            if w2i > 0:
                w2.append(2 * np.pi / np.sqrt(w2i))
            else:
                w2.append(float("nan"))  # critically or over-damped. There is no period
        return w2


class Force:
    """A driving force in 3 dimensions which produces an ouput per time and can be connected to the oscillator.

    Args:
        func (callable)=lambda t:np.array( (0,0,0), float): A function of t, producing a 3D vector
    """

    def __init__(self, func: Callable):
        self.func = func
        self.out = np.array((0, 0, 0), float)

    def do_step(self, current_time: float, step_size: float):
        self.out = self.func(current_time)
