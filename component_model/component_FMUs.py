"""Contains a few example FMU classes which can be build through this package
Note that they all need instantiation through derived classes with all arguments specified
"""

# from math import radians
import numpy as np
from .model import Model, ModelInitError, ModelOperationError
from .variable import Variable, Variable_NP


class InputTable(Model):
    """Simple input table component model.
    This serves also as an example how the component_model package can be used

    Args:
        name (str): a specific name of this input table
        description (str): a description of the output this input table delivers
        table (tuple): the table (as tuple of tuples) with time values in column 0 and any number of outputs
        outputName (str)='out': Optional possibility to specify the (parent) name of the outputs. Default: out.n with n=1,2,...
        interpolate (bool)=False: Optional possibility to interpolate between rows in the table at evry communication point,
          instead of discrete changes after the next time value is reached.
        kwargs: any argument of the Model class can be overridden through this mechanism

    """

    def __init__(
        self,
        name: str,
        description: str,
        author="Siegfried Eisinger",
        version="0.1",
        table: tuple = [[]],
        outputName: str = "out",
        interpolate: bool = False,
        **kwargs,
    ):
        super().__init__(
            name=name,
            description="At given times (specified in column 0 of the table a new vector of output values is generated and made available on output",
            author=author,
            version=version,
            **kwargs,
        )
        self._rows = len(table)
        self._cols = len(table[0]) - 1
        self.times = np.array(
            list(row[0] for row in table), dtype="float64"
        )  # this is only internally defined, not as Variable
        if not all(
            self.times[i - 1] < self.times[i] for i in range(1, len(self.times))
        ):
            raise ModelInitError(
                "The times in the input table are not properly sorted in ascending order"
            )
        if not self.times[0] == 0.0:
            raise ModelInitError(
                "The initial output values are not defined. The first time point should be at zero"
            )
        self.outputs = np.array(
            list(row[1:] for row in table), dtype="float64"
        )  # this is only internally defined, not as Variable
        self.ranges = self.get_ranges(self.outputs)
        self.outputName = outputName
        self.outs = Variable_NP(
            self,
            name="outputs",
            description="Output connector providing new outputs at every communication point (interpolation) or after the time of the next row is reached",
            causality="output",
            variability="continuous",
            initialVal=table[0][1:],
            rng=self.ranges,
        )
        self.interpolate = Variable(
            self,
            name="interpolate",
            description="Perform interpolation of outputs between time rows (instead of discrete outputs at communication point when next time is reached)",
            causality="parameter",
            variability="fixed",
            typ=bool,
            initialVal=interpolate,
        )
        self.time_iter = self.time_iterator()

    def do_step(self, currentTime, stepSize):
        """Do a simulation step of size 'stepSize at time 'currentTime"""
        # super().do_step( currentTime, stepSize) # this is not called here, because there are no on_set and other general issues
        self.set_values(currentTime)
        return True

    def time_iterator(self):
        """Define an iterator, which returns the next row reference and the time interval from the table"""
        for self.row in range(self._rows - 1):
            self.tInterval = (self.times[self.row], self.times[self.row + 1])
            yield self.row, self.tInterval
        yield self.row, self.tInterval  # stop at the last interval

    def set_values(self, time):
        """Retrieve the vector of values for the given time"""
        if time < self.tInterval[0]:
            raise ModelOperationError(
                f"set_values: Search time is before the current interval starting at {self.tInterval[0]}"
            )
        if time >= self.times[-1]:  # extrapolation
            print("COMPONENT_FMU 0", time, self.row, self.tInterval)
            self.tInterval = (self.times[-2], self.times[-1])
            self.row = self._rows - 2
        else:
            while (
                not self.tInterval[0] <= time < self.tInterval[1]
            ):  # move to the correct interval
                next(self.time_iter)
        print("COMPONENT_FMU", time, self.row, self.tInterval)
        if (
            not self.interpolate.value
        ):  # discrete values. Set the output values to the row values
            self.outs.value = self.outputs[self.row]
        else:
            dT = self.tInterval[1] - self.tInterval[0]
            for c in range(self._cols):
                dY = self.outputs[self.row + 1][c] - self.outputs[self.row][c]
                self.outs.value[c] = self.outputs[self.row][c] + dY / dT * (
                    time - self.tInterval[0]
                )

    #        print("COMPONENT_FMU", time, self.outs.value[0], self.outs.value[1], self.outs.value[2])

    def enter_initialization_mode(self):
        """Set the output values according to the start time. Ensure that always outputs are available for connected FMUs"""
        next(self.time_iter)  # move to row 0
        self.set_values(self.startTime)

    def get_ranges(self, tbl):
        """Calculate ranges of variables from the provided input table (min,max) per column"""
        ranges = []
        for c in range(len(tbl[0])):
            ranges.append((np.min(tbl[:, c]), np.max(tbl[:, c])))
        return ranges
