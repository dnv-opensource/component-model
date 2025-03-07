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


def func(time: float, ampl: float = 1.0, omega: float = 0.1):
    return np.array((0, 0, ampl * sin(omega * time)), float)


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
        func: Callable[..., Any] = func,
        ampl: float = 1.0,
        freq: float = 1.0,
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
        self._f = Variable(
            self,
            "f",
            "Output connector for the driving force f(t) in N",
            causality="output",
            variability="continuous",
            start=np.array((0, 0, 0), float),
        )

    def do_step(self, time: float, dt: float):
        self.f = self.func(time)
        return True  # very important!

    def exit_initialization_mode(self):
        """Set internal state after initial variables are set."""
        self.func = partial(func, ampl=self.ampl, omega=2 * pi * self.freq)  # type: ignore[reportAttributeAccessIssue]
        logger.info(f"Initial settings: ampl={self.ampl}, freq={self.freq}")  # type: ignore[reportAttributeAccessIssue]
