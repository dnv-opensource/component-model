import logging
from typing import Any

import numpy as np

from component_model.model import Model
from component_model.variable import Variable
from examples.oscillator import Oscillator

logger = logging.getLogger(__name__)


class HarmonicOscillator6D(Model, Oscillator):  # refer to Model first!
    """General harmonic oscillator with 6 DoF, extended for FMU packaging.

    The system obeys the equations
    F(t) - k*x - c*dx/dt = m*d^2x/dt^2  (first 3 equations, one per linear dimension)
    with F: external force, x: displacement and other symbols and units as defined below.
    See also `Harmonic_oscillator <https://en.wikipedia.org/wiki/Harmonic_oscillator>`_
    and
    T(t) - k*a - c*da/dt = I* d^a2/dt^2
    with T: external torque, a: deflection angle and other symbols and units as defined below.
    See also `Torsion_spring <https://en.wikipedia.org/wiki/Torsion_spring#Torsion_balance>`_
    Torque, the angles and all parameters are in cartesian pseudo vectors,
    i.e. a torque in z-direction implies a right-handed turn in the x-y plane.
    All parameters may vary along the 3 spatial dimensions.

    See also test_oscillator_fmu() where build process and further testing is found.
    For details see oscillator.py and test_oscillator.py, where the basic model is defined and tested.

    Args:
        k (tuple[float] | tuple[str])=(1.0,)*6: spring/torsion constants in N/m or N.m/rad
        c (tuple[float] | tuple[str])=(0.0,)*6: Viscous damping coefficients in N.s/m or N.m.s/rad
        m (float)=1.0: Mass of the spring load (spring mass negligible) in kg
        mi (tuple[float] | tuple[str])=(1.0,)*3: Moments of inertia in kg.m^2
        tolerance (float)=1e-5: Optional tolerance in m, i.e. maximum uncertainty in displacement x.
        x0 (tuple) = (0.0)*6: Start position in m or rad
        v0 (tuple) = (0.0)*6: Start speed in m/s or rad/s
    """

    def __init__(
        self,
        k: tuple[float, ...] | tuple[str, ...] = (1.0,) * 6,
        c: tuple[float, ...] | tuple[str, ...] = (0.1,) * 6,
        m: float | str = 1.0,
        mi: tuple[float, ...] | tuple[str, ...] = (1.0,) * 3,
        tolerance: float = 1e-5,
        x0: tuple[float, ...] | tuple[str, ...] = (0.0,) * 6,
        v0: tuple[float, ...] | tuple[str, ...] = (0.0,) * 6,
        **kwargs: Any,
    ):
        Model.__init__(
            self,  # here we define a few standard entries for FMU
            name="HarmonicOscillator",
            description="3D harmonic oscillator prepared for FMU packaging. 3D driving force can be connected",
            author="Siegfried Eisinger",
            version="0.2",
            default_experiment={"startTime": 0.0, "stopTime": 10.0, "stepSize": 0.01, "tolerance": 1e-5},
            **kwargs,
        )
        # include arguments to get the dimension right!
        Oscillator.__init__(self, (1.0,) * 6, (0.1,) * 6, (1.0,) * 6, tolerance=1e-3, f_func=None)
        # interface Variables.
        # Note that the Variable object is accessible as self._<name>, while the value is self.<name>
        self._k = Variable(self, "k", "The 6D spring constant in N/m or N.m/rad", start=k)
        self._c = Variable(self, "c", "The 6D spring damping in in N.s/m or N.m.s/rad", start=c)
        self._m = Variable(self, "m", "The mass at end of spring in kg", start=(m, m, m, mi[0], mi[1], mi[2]))
        self._x = Variable(
            self,
            "x",
            "The time-dependent 6D generalized position of the mass in m or rad",
            causality="output",
            variability="continuous",
            initial="exact",
            start=x0,
        )
        self._v = Variable(
            self,
            "v",
            "The time-dependent 6D generalized speed of the mass in m/s or rad/s",
            causality="output",
            variability="continuous",
            initial="exact",
            start=v0,
        )
        self._f = Variable(
            self,
            "f",
            "Input connector for the 6D external force acting on the mass in N or N.m",
            causality="input",
            variability="continuous",
            start=np.array((0,) * 6, float),
        )

    def do_step(self, current_time: float, step_size: float):
        if not Model.do_step(self, current_time, step_size):  # some housekeeping functions (not really needed here)
            return False
        Oscillator.do_step(self, current_time, step_size)  # this does the integration itself
        return True  # very important for the FMU mechanism

    def exit_initialization_mode(self):
        """Set internal state after initial variables are set.

        Note: need to update the ODE matrix to reflect changes in c, k or m!
        """
        self.ode = [
            np.array(((-self.c[i] / self.m[i], -self.k[i] / self.m[i]), (1, 0)), float) for i in range(self.dim)
        ]
        logger.info(f"Initial settings: k={self.k}, c={self.c}, m={self.m}, x={self.x}, v={self.v}, f={self.f}")

    # Note: The other FMU functions like .setup_experiment and  .exit_initialization_mode
    #       do not need special attention here and can be left out
