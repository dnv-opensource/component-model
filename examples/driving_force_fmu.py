import logging
from collections.abc import Callable
from functools import partial
from math import pi, sin
from typing import Any

import numpy as np

from component_model.model import Model
from component_model.variable import Variable

logger = logging.getLogger(__name__)


# Note: PythonFMU (which component-model is built on) works on files and thus only one Model class allowed per file


def func(time: float, ampl: float = 1.0, omega: float = 0.1, d_omega: float = 0.0):
    """Generate a harmonic oscillating force function.
    Optionally it is possible to linearly change the angular frequency as omega + d_omega*time.
    """
    if d_omega == 0.0:
        return np.array((0, 0, ampl * sin(omega * time)), float)
    else:           
        return np.array((0, 0, ampl * sin((omega + d_omega*time) * time)), float)


class DrivingForce(Model):
    """A driving force in 3 dimensions which produces an ouput per time and can be connected to the oscillator.

    Note: This completely replaces DrivingForce (do_step and other functions are not re-used).

    Args:
        func (callable)=func: The driving force function f(t).
            Note: The func can currently not really be handled as parameter and must be hard-coded here (see above).
            Soon to come: Model.build() function which honors parameters, such that function can be supplied from
            outside and the FMU can be re-build without changing the class.
    """

    def __init__(
        self,
        function: Callable[..., Any] = func,
        ampl: float = 1.0,
        freq: float = 1.0,
        d_freq: float = 0.0,
        **kwargs: Any,
    ):
        super().__init__(
            "DrivingForce",
            "A simple driving force for an oscillator",
            "Siegfried Eisinger",
            **kwargs,
        )
        # interface Variables
        self._ampl = Variable(self, "ampl", "The amplitude of the force in N", start=ampl)
        self._freq = Variable(self, "freq", "The frequency of the force in 1/s", start=freq)
        self._d_freq = Variable(self, "d_freq", "Change of frequency of the force in 1/s**2", start=d_freq)
        self.function = function
        self._f = Variable(
            self,
            "f",
            "Output connector for the driving force f(t) in N",
            causality="output",
            variability="continuous",
            start=np.array((0, 0, 0), float),
        )

    def do_step(self, current_time: float, step_size: float):
        self.f = self.func(current_time+step_size)
        return True  # very important!

    def exit_initialization_mode(self):
        """Set internal state after initial variables are set."""
        self.func = partial(self.function,
                            ampl=self.ampl,
                            omega=2 * pi * self.freq,
                            d_omega= 0.0 if self.d_freq == 0.0 else 2 * pi * self.d_freq)
        logger.info(f"Initial settings: ampl={self.ampl}, freq={self.freq}, d_freq={self.d_freq}")
