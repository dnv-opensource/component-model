import logging
from typing import Any

from component_model.model import Model
from component_model.variable import Variable
from examples.axle import Axle

logger = logging.getLogger(__name__)


class AxleFMU(Model, Axle):  # refer to Model first!
    """FMU packaging of Axle class object.

    Demonstrates also the use of structured FMI variables.

    Args:
        d1 (float)= 1.0: Diameter of wheel1 of the axle (parameter)
        rpm1 (float)= 1.0: Angle-speed of wheel2 (revolvements per second) of wheel1 (input)
        d2 (float)= 1.0: Diameter of wheel2 of the axle (parameter)
        rpm2 (float)= 1.0: Angle-speed (revolvements per second) of wheel2 (input)
        a (float)= 1.0: axle length of the axle (parameter)
    """

    def __init__(
        self,
        d1: float = 1.0,
        rpm1: float = -1.0,
        d2: float = 1.0,
        rpm2: float = -1.0,
        a: float = 1.0,
        **kwargs: Any,
    ):
        Model.__init__(
            self,  # here we define a few standard entries for FMU
            name="AxleFMU",
            description="Two wheels mounted on a fixed axle, driving with rpm as input.",
            author="Siegfried Eisinger",
            version="0.1",
            default_experiment={"startTime": 0.0, "stopTime": 10.0, "stepSize": 0.01, "tolerance": 1e-5},
            **kwargs,
        )
        Axle.__init__(self, d1, rpm1, d2, rpm2, a)
        self._a = Variable(
            self, "a", "Length of axle in m", causality="parameter", variability="fixed", start=f"{a} m", rng=(0, None)
        )
        self._d0 = Variable(
            self,
            "wheels[0].diameter",
            "Diameter of wheel 1 in m",
            causality="parameter",
            variability="fixed",
            start=f"{d1} m",
            rng=(0, None),
        )
        self._d1 = Variable(
            self,
            "wheels[1].diameter",
            "Diameter of wheel 1 in m",
            causality="parameter",
            variability="fixed",
            start=f"{d2} m",
            rng=(0, None),
        )
        self._rpm0 = Variable(
            self,
            "wheels[0].motor.rpm",
            "Angualar speed of wheel 1 in rad/s:",
            causality="input",
            variability="continuous",
            start=f"{rpm1} 1/s",
        )
        self._rpm1 = Variable(
            self,
            "wheels[1].motor.rpm",
            "Angualar speed of wheel 2 in rad/s:",
            causality="input",
            variability="continuous",
            start=f"{rpm2} 1/s",
        )
        self._acc0 = Variable(
            self,
            "der(wheels[0].motor.rpm)",
            "Angualar acceleration of wheel 1 in rad/s**2:",
            causality="input",
            variability="continuous",
            start="0.0 1/s**2",
            local_name="acc",
        )
        self._acc1 = (
            Variable(
                self,
                "der(wheels[1].motor.rpm)",
                "Angualar acceleration of wheel 2 in rad/s**2:",
                causality="input",
                variability="continuous",
                start="0.0 1/s**2",
            ),
        )
        self._pos0 = Variable(
            self,
            "wheels[0].pos",
            "2D position of wheel 1 in [m,m]",
            causality="output",
            variability="continuous",
            initial="exact",
            start=("0.0 m", "0.0 m"),
        )
        self._pos1 = Variable(
            self,
            "wheels[1].pos",
            "2D position of wheel 2 in [m,m]",
            causality="output",
            variability="continuous",
            initial="exact",
            start=("0.0 m", f"{a} m"),
        )

    def do_step(self, current_time: float, step_size: float):
        Model.do_step(self, current_time, step_size)  # some housekeeping functions (not really needed here)
        Axle.drive(self, current_time, step_size)  # this does the integration itself
        return True  # very important for the FMU mechanism

    def exit_initialization_mode(self):
        """Set internal state after initial variables are set."""
        self.init_drive()
