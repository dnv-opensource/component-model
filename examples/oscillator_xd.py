from typing import Any, Callable

import numpy as np
from scipy import integrate


class OscillatorXD:
    """Construct a simple model of a general harmonic oscillator, potentially driven by a force.

    The system obeys the equation F(t) - k*x - c*dx/dt = m*d^2x/dt^2
    See also `Wikipedia <https://en.wikipedia.org/wiki/Harmonic_oscillator>`_

    where x shall be an xD vector with an initial position and velocity. Common dimentsions are

    * 1: a one dimensional oscillator (e.g. a pendulum)
    * 3: a three-dimensional oscillator in 3D space
    * 6: a six-dimensional oscillator, represented e.g. by a rigid body mounted on springs.
      The generalized position is in this case the 3D cartesian position + 3D angular position

    F(t)=0 as long as there is not external driving force.
    NOTE: This is a basic oscillator model.

    * Unrelated with Model and Variable
    * FMU cannot be built from that! See oscillator_fmu where this is extended to an FMU package.
    * The HarmonicOscillator can be used for model testing (highly recomended, see test_oscillator_xv.py)
    * It can of cause be used stand-alone

    We use the scipy.integrate.solve_ivp algorithm for the integration (see do_step)

    Args:
        dim (int)=6: dimension of the oscillator
        k (float|tuple)=1.0: spring constant in N/m (if x position). May vary in the dimensions if tuple
        c (float|tuple)=0.0: Viscous damping coefficient in N.s/m (if x position). May vary in the dimensions if tuple
        m (float | tuple)=1.0: Mass of the spring load (spring mass negligible) in kg if x position.
          Same in all dim if float.
        tolerance (float)=1e-5: Optional tolerance in m, i.e. maximum uncertainty in displacement x.
        force (obj): Force object
    """

    def __init__(
        self,
        dim: int = 6,
        k: float | tuple[float, ...] | tuple[str, ...] = 1.0,
        c: float | tuple[float, ...] | tuple[str, ...] = 0.0,
        m: float | tuple[float, ...] = 1.0,
        tolerance: float = 1e-5,
        force: Any | None = None,
    ):
        self.dim = dim
        self._m = np.array((m,) * self.dim, float) if isinstance(m, float) else np.array(m, float)
        assert len(self._m) == self.dim, f"Expect dimension {self.dim} for mass. Found: {self._m}"

        if isinstance(k, float):
            self.w2 = np.array((k,) * self.dim, float) / self._m
        elif isinstance(k, tuple):
            assert len(k) == self.dim, f"Expect dimension {self.dim} for k. Found: {k}"
            self.w2 = np.array(k, float) / self._m
        else:
            raise ValueError(f"Unhandled type for 'k': {k}")

        if isinstance(c, float):
            self.gam = np.array((c / 2.0,) * self.dim, float) / self._m
        elif isinstance(c, tuple):
            assert len(c) == self.dim, f"Expect dimension {self.dim} for k. Found: {k}"
            self.gam = np.array(c, float) / self._m / 2.0
        else:
            raise ValueError(f"Unhandled type of 'c': {c}") from None

        self.tolerance = tolerance
        self.x = np.array((0,) * 2 * self.dim, float)  # generalized position + generalized first derivatives
        self.v = self.x[self.dim :]  # link to generalized derivatives (for convenience)
        self.force = force  # the function object

    def ode_func(
        self,
        t: float,  # scalar time
        y: np.ndarray,  # combined array of positions and speeds
    ) -> np.ndarray:  # derivative of positions and speeds
        if self.force is None:
            return np.append(y[self.dim :], -2.0 * self.gam * y[self.dim :] - self.w2 * y[: self.dim])
        else:  # explicit force function is defined
            f = self.force(t=t, x=y[: self.dim], v=y[self.dim :])
            d_dt = -2.0 * self.gam * y[self.dim :] - self.w2 * y[: self.dim] + f
            # print(f"ODE({self.dim})@{t}. f:{f[2]}, z:{y[2]}, v:{y[8]} => d_dt:{d_dt[2]}.")
            return np.append(y[self.dim :], d_dt)

    def do_step(self, current_time: float, step_size: int | float) -> bool:
        """Do one simulation step of size dt.

        We implement a very simplistic algoritm based on difference calculus.
        """
        sol = integrate.solve_ivp(
            fun=self.ode_func,
            t_span=[current_time, current_time + step_size],
            y0=self.x,  # use the combined position-velocity array
            atol=self.tolerance,
        )
        self.x = sol.y[:, -1]
        self.v = self.x[self.dim :]  # re-link
        return True  # to keep the signature when moving to FMU

    @property
    def period(self):
        """Calculate the natural period of the oscillator (without damping)."""
        w2 = []
        for i in range(self.dim):
            w2i = self.w2[i] - self.gam[i] ** 2
            if w2i > 0:
                w2.append(2 * np.pi / np.sqrt(w2i))
            else:
                w2.append(float("nan"))  # critically or over-damped. There is no period
        return w2


class Force:
    """A driving force in x dimensions which produces an ouput as function of time and/or position.
    Can be connected to the oscillator.

    Args:
        dim (int): the dimension of the force (6: 3*linear force + 3*torque
        func (callable): Function of agreed arguments (see args)
        _args (str): Sequence of 't' (time), 'x' (position), 'v' (velocity),
          denoting the force dependencies
    """

    def __init__(self, dim: int, func: Callable):
        self.dim = dim
        self.func = func
        self.current_time = 0.0
        self.dt = 0
        self.out = np.array((0,) * self.dim)

    def __call__(self, **kwargs):
        """Calculate the force in dependence on keyword arguments 't', 'x' or 'v'."""
        if "t" in kwargs:
            t = kwargs["t"]
            if abs(t - self.current_time) < 1e-15:
                return self.out
            self.dt = t - self.current_time
            self.current_time = t
            kwargs.update({"dt": self.dt})  # make the step size known to the function
        else:
            kwargs.update({"t": None, "dt": None})
        self.out = self.func(**kwargs)
        return self.out
