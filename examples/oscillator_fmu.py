from typing import Any

import numpy as np

from component_model.model import Model
from component_model.variable import Variable
from examples.oscillator import HarmonicOscillator


class HarmonicOscillatorFMU(Model, HarmonicOscillator):  # refer to Model first!
    """General 3D harmonic oscillator as defined in HarminicOscillator, extended for FMU packaging.

    The system obeys the equation F(t) - k/m*x - c/m*dx/dt = d^2x/dt^2  (3 equations, one per dimension)
    See also `Wikipedia <https://en.wikipedia.org/wiki/Harmonic_oscillator>`_

    NOTE: This is FMU extension, demonstrating how to define the interface and prepare for building the FMU.

    See also test_oscillator_fmu() where build process and further testing is found.
    For details see oscillator.py and test_oscillator.py.

    Args:
        k (tuple[float] | tuple[str])=(1.0, 1.0, 1.0): spring constant in N/m. May vary in 3D
        c (tuple[float] | tuple[str])=(0.0, 0.0, 0.0): Viscous damping coefficient in N.s/m. May vary in 3D
        m (float)=1.0: Mass of the spring load (spring mass negligible) in kg
        tolerance (float)=1e-5: Optional tolerance in m, i.e. maximum uncertainty in displacement x.
    """

    def __init__(
        self,
        k: tuple[float, float, float] | tuple[str, str, str] = (1.0, 1.0, 1.0),
        c: tuple[float, float, float] | tuple[str, str, str] = (0.0, 0.0, 0.0),
        m: float | str = 1.0,
        tolerance: float = 1e-5,
        x0: tuple[float, float, float] | tuple[str, str, str] = (1.0, 1.0, 1.0),
        v0: tuple[float, float, float] | tuple[str, str, str] = (0.0, 0.0, 0.0),
        **kwargs: Any,
    ):
        super().__init__(  # here we define a few standard entries for FMU
            name="3D oscillator",
            description="3D harmonic oscillator prepared for FMU packaging. 3D driving force can be connected",
            author="Siegfried Eisinger",
            version="0.2",
            default_experiment={"startTime": 0.0, "stopTime": 10.0, "stepSize": 0.01, "tolerance": 1e-5},
            **kwargs,
        )
        # interface Variables.
        # Note that the Variable object is accessible as self._<name>, while the value is self.<name>
        self._k = Variable(self, "k", "The 3D spring constant in N/m", start=k)
        self._c = Variable(self, "c", "The 3D spring damping in in N.s/m", start=c)
        self._m = Variable(self, "m", "The mass at end of spring in kg", start=m)
        self._x = Variable(
            self,
            "x",
            "The time-dependent 3D position of the mass in m",
            causality="output",
            variability="continuous",
            initial="exact",
            start=x0,
        )
        self._v = Variable(
            self,
            "v",
            "The time-dependent 3D speed of the mass in m/s",
            causality="output",
            variability="continuous",
            initial="exact",
            start=v0,
        )
        self._f = Variable(
            self,
            "f",
            "Input connector for the 3D external force acting on the mass in N",
            causality="input",
            variability="continuous",
            start=np.array((0, 0, 0), float),
        )

    def do_step(self, time: float, dt: float):
        super().do_step(time, dt)  # this does the integration itself
        return True  # very important for the FMU mechanism

    def exit_initialization_mode(self):
        """Set internal state after initial variables are set."""
        print(f"Initial settings: k={self.k}, c={self.c}, m={self.m}, x={self.x}, v={self.v}, f={self.f}")

    # Note: The other FMU functions like .setup_experiment and  .exit_initialization_mode
    #       do not need special attention here and can be left out
