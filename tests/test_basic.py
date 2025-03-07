from pathlib import Path

import pytest
from fmpy import simulate_fmu  # type: ignore
from pythonfmu import (  # type: ignore
    Boolean,
    DefaultExperiment,
    Fmi2Causality,
    Fmi2Slave,
    Fmi2Variability,
    FmuBuilder,
    Integer,
    Real,
    String,
)


@pytest.fixture(scope="package")
def build_fmu():
    build_path = Path.cwd()
    build_path.mkdir(exist_ok=True)
    fmu_path = FmuBuilder.build_FMU(__file__, project_files=[], dest=build_path)
    return fmu_path


class PythonSlave(Fmi2Slave):
    author = "John Doe"
    description = "A simple description"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.intOut = 1
        self.realOut = 3.0
        self.realIn = 99.9
        self.booleanVariable = True
        self.stringVariable = "Hello World!"
        self.realConst = -1.0
        self.register_variable(
            Integer(
                "intOut",
                causality=Fmi2Causality.output,
                variability=Fmi2Variability.discrete,
            )
        )
        self.register_variable(
            Real(
                "realOut",
                causality=Fmi2Causality.output,
                variability=Fmi2Variability.continuous,
            )
        )
        self.register_variable(
            Boolean(
                "booleanVariable",
                causality=Fmi2Causality.local,
                variability=Fmi2Variability.discrete,
            )
        )
        self.register_variable(
            String(
                "stringVariable",
                causality=Fmi2Causality.local,
                variability=Fmi2Variability.discrete,
            )
        )
        self.register_variable(
            Real(
                "realIn",
                causality=Fmi2Causality.input,
                variability=Fmi2Variability.continuous,
            )
        )
        self.register_variable(
            Real(
                "realConst",
                causality=Fmi2Causality.parameter,
                variability=Fmi2Variability.fixed,
            )
        )

        self.default_experiment = DefaultExperiment(start_time=0.0, stop_time=1.0, step_size=0.1)

        # Note:
        # it is also possible to explicitly define getters and setters as lambdas in case the variable is not backed by a Python field.
        # self.register_variable(Real("myReal", causality=Fmi2Causality.output, getter=lambda: self.realOut, setter=lambda v: set_real_out(v))

    def setup_experiment(self, start_time: float):
        """1. After instantiation the expriment is set up. In addition to start and end time also constant input variables are set."""
        assert [self.vars[idx].getter() for idx in range(5)] == [
            1,
            3.0,
            True,
            "Hello World!",
            99.9,
        ], "Values of first 5 instantiated variables"

    def enter_initialization_mode(self):
        """2.  During initialization all other input variables are set."""
        assert [self.vars[idx].getter() for idx in range(5)] == [
            1,
            3.0,
            True,
            "Hello World!",
            88.8,
        ], "Start values of first 5 variables"

    def exit_initialization_mode(self):
        pass

    def do_step(self, current_time, step_size):
        """N. Happens at every communication point.
        a. Inputs (signals) are set
        b. Perform calculations
        c. Outputs are retrieved
        """
        #        self.vars[1].setter( self.vars[1].getter()+1)
        self.realIn = self.realOut
        assert self.realOut == 3.0 + current_time * 10, "Value of 'realOut' during run"
        self.realOut += 1
        return True


def test_make_fmu(build_fmu):
    assert build_fmu.name == "PythonSlave.fmu"


def test_use_fmu(build_fmu):
    _ = simulate_fmu(  # type: ignore #fmpy does not comply to pyright expectations
        build_fmu,
        stop_time=1,
        step_size=0.1,
        validate=True,
        solver="Euler",
        debug_logging=True,
        logger=print,  # fmi_call_logger=print,
        start_values={"realIn": 88.8},
    )


if __name__ == "__main__":
    retcode = pytest.main(["-rA", "-v", __file__])
    assert retcode == 0, f"Non-zero return code {retcode}"
