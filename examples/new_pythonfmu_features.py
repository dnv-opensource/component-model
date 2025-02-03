from typing import Any

from pythonfmu.enums import Fmi2Status

from component_model import Model, Variable


class NewFeatures(Model):
    """Dummy model to test new features of component-model and pythonfmu.

    * logger messages to the user
    + handle __init__ parameters
    + translate assert statements to logger messages
    + translate Exceptions to logger messages
    + allow class as argument to .build, instead of the source code file
    """

    def __init__(
        self,
        i: int = 1,
        f: float = 9.9,
        s: str = "Hello",
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        super().__init__(
            "NewFeatures",
            "Dummy model for testing new features in PythonFMU",
            "Siegfried Eisinger",
            default_experiment={"startTime": 0, "stopTime": 9, "stepSize": 1},
            **kwargs,
        )
        print("NAME", self, self.name)
        self._i = Variable(
            self,
            "i",
            "My integer",
            typ=int,
            causality="output",
            variability="discrete",
            initial="exact",
            start=i,
            rng=(0, 10),
        )

        self._f = Variable(self, "f", "My float", causality="input", variability="continuous", start=f)

        self._s = Variable(self, "s", "My string", typ=str, causality="parameter", variability="fixed", start=s)

        self.log("This is a __init__ debug message", debug=True)
        # self.log("This is a FATAL __init__ message", status=Fmi2Status.fatal, category="logStatusFatal", debug=False)  # noqa: ERA001

    def do_step(self, time: int | float, dt: int | float):
        super().do_step(time, dt)
        self.i += 1
        self.f = time
        assert self.f == time  # assert without message, but comment
        # assert self.i < 8, "The range check would detect that with the next get message"
        # send log messages of all types. OSP makes them visible according to log_output_level setting
        self.log(f"do_step@{time}. logAll", status=Fmi2Status.ok, category="logAll", debug=True)
        self.log(
            f"do_step@{time}. logStatusWarning", status=Fmi2Status.warning, category="logStatusWarning", debug=True
        )
        self.log(
            f"do_step@{time}. logStatusDiscard", status=Fmi2Status.discard, category="logStatusDiscard", debug=True
        )
        self.log(f"do_step@{time}. logStatusError", status=Fmi2Status.error, category="logStatusError", debug=True)
        # self.log(f"do_step@{time}. logStatusFatal", status=Fmi2Status.fatal, category="logStatusFatal", debug=True)  # noqa: ERA001
        if time > 8:
            self.log(
                f"@{time}. Trying to terminate simulation",
                status=Fmi2Status.error,
                category="logStatusError",
                debug=True,
            )
            return False
        return True

    def exit_initialization_mode(self):
        print(f"My initial variables: i:{self.i}, f:{self.f}, s:{self.s}")
