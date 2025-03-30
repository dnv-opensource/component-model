import logging
from typing import Sequence

from component_model.model import Model
from component_model.variable import Variable
from examples.time_table import TimeTable

logger = logging.getLogger(__name__)


class TimeTableFMU(Model, TimeTable):  # refer to Model first!
    """FMU wrapper of LookupTable model class.

    Simple lookup table component model.
    Pre-loaded with data it can be used for lookup or interpolation, e.g. as surrogate model.
    Note: currently the data in the table is fixed inside the FMU and the FMU must be re-built to change the table,
       which is due to the fact that support for tables is so far not included in the component-model package.

    Args:
        name (str): a specific name of this lookup table
        description (str): a description of the output this lookup table delivers
        data (iterable): the table (as tuple of tuples) with time values in column 0 and any number of outputs
        header (list): Optional possibility to specify the name of the outputs. Default: out.n with n=1,2,...
        interpolate (int)=1: Set interpolation level. 0: discrete, 1: linear, 2: quadratic, 3: cubic.
        kwargs: any argument of the Model class can be overridden through this mechanism
    """

    def __init__(
        self,
        data: Sequence = ((0.0, 1, 0, 0), (1.0, 1, 1, 1), (3.0, 1, 3, 9), (7.0, 1, 7, 49)),  # data useful for testing
        header: Sequence[str] | None = None,
        interpolate: int = 1,
        default_experiment: dict[str, float] | None = None,
        **kwargs,
    ):
        TimeTable.__init__(self, data, header, interpolate)
        if default_experiment is None:
            default_experiment = {"startTime": 0, "stopTime": 10.0, "stepSize": 0.1, "tolerance": 1e-5}
        Model.__init__(
            self,  # here we define a few standard entries for FMU
            name="TimeTable",
            description="Simple time-based lookup table for use in simulations (e.g. surrogate models)",
            author="Siegfried Eisinger",
            version="0.1",
            default_experiment=default_experiment,
            **kwargs,
        )
        # Note that the Variable object is accessible as self._<name>, while the value is self.<name>
        self._interpolate = Variable(
            self,
            "interpolate",
            "The interpolation exponent. 0: discrete ... 3: cubic",
            typ=int,
            causality="parameter",
            variability="fixed",
            initial="exact",
            start=interpolate,
            rng=(0, 4),
            on_set=self.set_interpolate,
        )  # the interpolation type can be set as parameter
        self._outs = Variable(
            self,
            "outs",
            "The lookup values for a given time",
            causality="output",
            variability="continuous",
            initial="exact",
            start=self.outs,
        )

    def do_step(self, current_time: float, step_size: float):
        Model.do_step(self, current_time, step_size)  # some housekeeping functions (not really needed here)
        TimeTable.lookup(self, current_time + step_size)  # this does the lookup itself
        return True  # very important for the FMU mechanism
