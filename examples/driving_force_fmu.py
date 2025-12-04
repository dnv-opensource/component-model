import logging
from functools import partial
from typing import Any, Callable

import numpy as np

from component_model.model import Model
from component_model.variable import Variable

logger = logging.getLogger(__name__)


# Note: PythonFMU (which component-model is built on) works on files and thus only one Model class allowed per file


def func(time: float, ampl: np.ndarray, omega: np.ndarray, d_omega: np.ndarray):
    """Generate a harmonic oscillating force function.
    Optionally it is possible to linearly change the angular frequency as omega + d_omega*time.
    The function is intended to be initialized through partial, so that only 'time' is left as variable.
    """
    if all(_do == 0.0 for _do in d_omega):
        return ampl * np.sin(omega * time)
    else:
        return ampl * np.sin((omega + d_omega * time) * time)


class DrivingForce(Model):
    """A driving force in 3 dimensions which produces an ouput per time and can be connected to the oscillator.

    Note1: the FMU model is made directly (without a basic python class model), which is not recommended!
    Note2: the speed of the connected oscillator is added as additional connector.
    Since the driving is forced, the input speed is ignored, but it is needed for the ECCO algorithm (power bonds).

    Note: This completely replaces DrivingForce (do_step and other functions are not re-used).

    Args:
        func (callable)=func: The driving force function f(t).
        ampl (float|tuple) = 1.0: the amplitude of the (sinusoidal) driving force. Same for all D if float.
          Optional with units.
        freq (float|tuple) = 1.0: the frequency of the (sinusoidal) driving force. Same for all D if float.
          Optional with units.
        d_freq (float) = 0.0: Optional frequency change per time unit (for frequency sweep experiments).
    """

    def __init__(
        self,
        ampl: float | tuple[float] | tuple[str] = 1.0,
        freq: float | tuple[float] | tuple[str] = 1.0,
        d_freq: float | tuple[float] | tuple[str] = 0.0,
        **kwargs: Any,
    ):
        super().__init__(
            "DrivingForce",
            "A simple driving force for an oscillator",
            "Siegfried Eisinger",
            **kwargs,
        )
        # interface Variables. We define first their values, to help pyright, since the basic model is missing
        _ampl = ampl if isinstance(ampl, tuple) else (ampl,)
        self.dim = len(_ampl)
        _freq = freq if isinstance(freq, tuple) else (freq,)
        assert len(_freq) == self.dim, f"ampl and freq are expected of same length. Found {ampl}, {freq}"
        _d_freq = d_freq if isinstance(d_freq, tuple) else (d_freq,) * self.dim
        assert len(_d_freq) == self.dim, f"d_freq expected as float or has same length as ampl:{ampl}. Found {d_freq}"
        self.ampl = np.array((1.0,) * self.dim, float)
        self.freq = np.array((1.0,) * self.dim, float)
        self.d_freq = np.array((0.0,) * self.dim, float)
        self.function = func
        self.func: Callable
        self.f = np.array((0.0,) * self.dim, float)
        self.v_osc = (0.0,) * self.dim
        self._ampl = Variable(self, "ampl", "The amplitude of the force in N", start=_ampl)
        self._freq = Variable(self, "freq", "The frequency of the force in 1/s", start=_freq)
        self._d_freq = Variable(self, "d_freq", "Change of frequency of the force in 1/s**2", start=_d_freq)
        self._f = Variable(
            self,
            "f",
            "Output connector for the driving force f(t) in N",
            causality="output",
            variability="continuous",
            start=(0.0,) * self.dim,
        )
        self._v_osc = Variable(
            self,
            "v_osc",
            "Input connector for the speed of the connected element in m/s",
            causality="input",
            variability="continuous",
            start=(0.0,) * self.dim,
        )

    def do_step(self, current_time: float, step_size: float):
        self.f = self.func(current_time + step_size)
        return True  # very important!

    def exit_initialization_mode(self):
        """Set internal state after initial variables are set."""
        assert isinstance(self.ampl, np.ndarray)
        assert isinstance(self.freq, np.ndarray)
        assert isinstance(self.d_freq, np.ndarray)

        self.func = partial(
            self.function,
            ampl=np.array(self.ampl, float),
            omega=np.array(2 * np.pi * self.freq, float),  # type: ignore # it is an ndarray!
            d_omega=np.array(2 * np.pi * self.d_freq, float),  # type: ignore # it is an ndarray!
        )
        logger.info(f"Initial settings: ampl={self.ampl}, freq={self.freq}, d_freq={self.d_freq}")
