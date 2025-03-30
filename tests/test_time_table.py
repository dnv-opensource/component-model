from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pytest
from examples.time_table import TimeTable


def plot(times: np.ndarray, tbl: TimeTable, col: int):
    x = [tbl.lookup(t)[col] for t in times]
    plt.plot(times, x, "-", label=f"interpolate{tbl.interpolate}[{col}]")
    plt.plot(tbl.times, tbl.data[:, col], "o", label="data")
    plt.show()


def arrays_equal(
    res: Iterable[Any],
    expected: Iterable[Any],
    eps: float = 1e-7,
):
    len_res = len(list(res))
    len_exp = len(list(expected))
    if len_res != len_exp:
        raise ValueError(f"Arrays of different lengths cannot be equal. Found {len_res} != {len_exp}")
    for i, (x, y) in enumerate(zip(res, expected, strict=False)):
        assert abs(x - y) < eps, f"Element {i} not nearly equal in {x}, {y}"


def test_time_table(show: bool = False):
    tbl = TimeTable(
        data=((0.0, 1, 0, 0), (1.0, 1, 1, 1), (3.0, 1, 3, 9), (7.0, 1, 7, 49)),
        header=("x", "y", "z"),
        interpolate=0,
    )
    assert not tbl.interpolate, f"Interpolation=0 expected. Found {tbl.interpolate}"
    assert all(tbl.times[i] == [0.0, 1.0, 3.0, 7.0][i] for i in range(tbl._rows)), (
        f"Expected time array [0,1,3,7]. Found {tbl.times}"
    )
    # print("DATA", tbl.data)
    # print("OUTS", tbl.outs[0])
    for r in range(tbl._rows):
        expected = [[1, 0, 0], [1, 1, 1], [1, 3, 9], [1, 7, 49]][r]
        arrays_equal(tbl.data[r], expected)
    arrays_equal(tbl.outs, (1, 0, 0))
    for r, time in enumerate(tbl.times):
        tbl.lookup(time)
        assert tbl.times[r] == time, f"Exact time {time}=={tbl.times[r]} at row {r}"
        assert all(tbl.outs[c] == tbl.data[r][c] for c in range(tbl._cols)), f"Outs at row {r}"
    for i in range(90):
        time = -1.0 + 0.1 * i
        tbl.lookup(time)
        if tbl.times[0] <= time <= tbl.times[-1]:  # no extrapolation
            if time <= tbl.times[0]:
                arrays_equal(tbl.outs, tbl.data[0, :])
            elif time >= tbl.times[-1]:
                arrays_equal(tbl.outs, tbl.data[-1, :])
            else:  # inside
                for i, t in enumerate(tbl.times):
                    if t > time:
                        arrays_equal(tbl.outs, tbl.data[i - 1, :])
                        break
    for t in np.linspace(0, 5, 100):
        assert tbl.lookup(t)[0] == 1.0

    tbl.set_interpolate(1)
    assert tbl.interpolate == 1, f"Interpolation=1 expected. Found {tbl.interpolate}"
    for t in np.linspace(0, 5, 100):
        assert abs(tbl.lookup(t)[1] - t) < 1e-10

    tbl.set_interpolate(2)
    assert tbl.interpolate == 2, f"Interpolation=2 expected. Found {tbl.interpolate}"
    if show:
        plot(np.linspace(0, 5, 100), tbl, 2)
    for t in np.linspace(0, 5, 100):
        assert abs(tbl.lookup(t)[2] - t**2) < 1e-10


if __name__ == "__main__":
    retcode = pytest.main(["-rA", "-v", __file__])
    assert retcode == 0, f"Non-zero return code {retcode}"
    import os

    os.chdir(Path(__file__).parent.absolute() / "test_working_directory")
    # test_time_table()
