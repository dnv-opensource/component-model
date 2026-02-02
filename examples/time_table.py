from typing import Any

import numpy as np
from scipy.interpolate import make_interp_spline


class TimeTable:
    """Simple lookup table component model.
    Pre-loaded with data it can be used for lookup or interpolation.

    To be used as FMU, the class must be extended with an interface.

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
        data: list[list[int | float]] | None = None,
        header: list[str] | None = None,
        interpolate: int = 1,
        **kwargs: Any,
    ):
        if data is None:
            data = [  # default data set useful for testing
                [0.0, 1, 0, 0],
                [1.0, 1, 1, 1],
                [3.0, 1, 3, 9],
                [7.0, 1, 7, 49],
            ]
        self._rows = len(data)
        assert self._rows > 0, "Empty lookup table detected, which does not make sense"
        self._cols = len(data[0]) - 1
        assert self._cols > 0, "No data column found in lookup table"
        self.times = np.array(list(row[0] for row in data), float)  # column 0 as times
        assert all(self.times[i - 1] < self.times[i] for i in range(1, len(self.times))), (
            "The times in the input data are not properly sorted in ascending order"
        )
        self.data = np.array(list(row[1:] for row in data), float)
        if header is None:
            self.header = tuple([f"out.{i}" for i in range(self._cols)])
        else:
            assert len(header) == self._cols, "Number of header elements does not match number of columns in data"
            self.header = tuple(header)
        self.outs = self.data[0]  # initial values
        self.interpolate = self.set_interpolate(int(interpolate))

    def set_interpolate(self, interpolate: int):
        assert 0 <= interpolate <= 4, f"Erroneous interpolation exponent {self.interpolate}"
        self._bspl = make_interp_spline(self.times, self.data, k=int(interpolate))
        self.interpolate = interpolate
        return interpolate

    def lookup(self, time: float):
        """Do a simulation step of size 'stepSize at time 'time."""
        self.outs = self._bspl(time)
        return self.outs
