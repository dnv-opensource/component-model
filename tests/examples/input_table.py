import numpy as np
from component_model.model import Model
from component_model.variable import Variable


class InputTable(Model):
    """Simple input table component model.
    This serves also as an example how the component_model package can be used.

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
        table: tuple = (),
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
        assert all(
            self.times[i - 1] < self.times[i] for i in range(1, len(self.times))
        ), "The times in the input table are not properly sorted in ascending order"
        assert self.times[0] == 0.0, "The initial output values are not defined. The first time point should be at zero"
        self.outputs = np.array(
            list(row[1:] for row in table), dtype="float64"
        )  # this is only internally defined, not as Variable
        self.outputName = outputName
        self._interface(table[0][1:], interpolate)

    def _interface(self, outs0: tuple, interpolate0: bool):
        self._outs = Variable(
            self,
            name="outs",
            description="Output connector providing new outputs at every communication point (interpolation) or after the time of the next row is reached",
            causality="output",
            variability="continuous",
            start=outs0,
            typ=float,
        )
        #        self.set_ranges( interpolate) # set the range separately, since it might change if 'interpolate' is changed
        self._interpolate = Variable(
            self,
            name="interpolate",
            description="Perform interpolation of outputs between time rows (instead of discrete outputs at communication point when next time is reached)",
            causality="parameter",
            variability="fixed",
            typ=bool,
            start=interpolate0,
            on_set=self.set_ranges,  # need to adapt ranges when 'interpolate' changes
        )

    def do_step(self, time, stepSize):
        """Do a simulation step of size 'stepSize at time 'time."""
        # super().do_step( time, stepSize) # this is not called here, because there are no on_set and other general issues
        self.set_values(time)
        return True

    @staticmethod
    def extrap(dt: float, tt, yy):
        """Calculate the extrapolated value using the provided first/last data set.
        dt<0 for left extrapolation, dt>0 for right extrapolation.
        """
        der = (yy[1] - yy[0]) / (tt[1] - tt[0])
        return (yy[0] if dt < 0 else yy[1]) + dt * der

    def set_values(self, time: float):
        """Retrieve the vector of values for the given time."""
        # discrete values. Set the output values to the row values
        if not self.interpolate:
            if time <= self.times[0]:
                self.outs = self.outputs[0]
            elif time >= self.times[-1]:
                self.outs = self.outputs[-1]
            else:
                for i, t in enumerate(self.times):
                    if t > time:
                        self.outs = self.outputs[i - 1]
                        break
        else:  # interpolate
            for c in range(self._cols):
                self.outs[c] = np.interp(
                    time,
                    self.times,
                    self.outputs[:, c],
                    left=InputTable.extrap(time - self.times[0], self.times[:2], self.outputs[:2, c]),
                    right=InputTable.extrap(time - self.times[-1], self.times[-2:], self.outputs[-2:, c]),
                )

    def set_ranges(self, interpolate: bool):
        """Set the ranges of 'outs' from the table (min,max) and the 'interpolate' setting per column."""
        ranges = []
        for c in range(len(self.outputs[0])):
            ranges.append([np.min(self.outputs[:, c]), np.max(self.outputs[:, c])])
            if interpolate:
                dleft = InputTable.extrap(-1.0, self.times[:2], self.outputs[:2, c]) - self.outputs[0, c]
                if dleft < 0:
                    ranges[c][0] = float("-inf")
                elif dleft > 0:
                    ranges[c][1] = float("inf")
                dright = InputTable.extrap(1.0, self.times[-2:], self.outputs[-2:, c]) - self.outputs[-1, c]
                if dright < 0:
                    ranges[c][0] = float("-inf")
                elif dright > 0:
                    ranges[c][1] = float("inf")
        self._outs.range = tuple((r[0], r[1]) for r in ranges)  # as tuples (not lists)
        return interpolate  # return that so that self.interpolate is set
