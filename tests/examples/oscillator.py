import numpy as np

from component_model.model import Model
from component_model.variable import Variable


class HarmonicOscillator(Model):
    """Construct a simple model of a general harmonic oscillator, potentially driven by a force.

    The system obeys the equation F(t) - k*x - c*dx/dt = m*d^2x/dt^2

    where x shall be a 3D vector with an initial position. F(t)=0 as long as there is not external driving force.

    Args:
        k (float)=1.0: spring constant in N/m
        c (float)=0.0: Viscous damping coefficient in N.s/m
        m (float)=1.0: Mass of the spring load (spring mass negligible) in kg

    See also `Wikipedia <https://en.wikipedia.org/wiki/Harmonic_oscillator>`_
    """

    def __init__(self, k: float = 1.0, c: float = 0.1, m: float = 1.0, **kwargs):
        super().__init__("Oscillator", "A simple harmonic oscillator", "Siegfried Eisinger", **kwargs)
        self.k = k
        self.c = c
        self.m = m
        self.x = (0.0, 0.0, 0.0)
        self.v = np.array((0, 0, 0), float)
        self.f = np.array((0, 0, 0), float)

    def do_step(self, time: float, dt: float) -> bool:
        """Do one simulation step of size dt.

        We implement a very simplistic algoritm based on difference calculus.
        """
        super().do_step(time, dt)  # needed for FMU mechanism
        a = (self.f - self.k * self.x - self.c * self.v) / self.m
        self.x += self.v * dt  # + a* dt*dt
        self.v += a * dt

        return True

    def setup_experiment(self, start):
        super().setup_experiment(start)  # needed for FMU mechanism


class DrivingForce(Model):
    """A driving force in 3 dimensions which produces an ouput per time and can be connected to the oscillator.

    Args:
        func (callable)=lambda t:np.array( (0,0,0), float): A function of t, producing a 3D vector
    """

    def __init__(self, func: callable):
        self.func = func
        self.out = np.array((0, 0, 0), float)

    def do_step(self, time: float, dt: float):
        self.out = self.func(time)
