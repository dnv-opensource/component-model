import xml.etree.ElementTree as ET  # noqa: N817
from zipfile import ZipFile

import numpy as np
from component_model.component_fmus import InputTable  # type: ignore
from component_model.model import Model # type: ignore
from fmpy import plot_result, simulate_fmu  # type: ignore
from fmpy.util import fmu_info  # type: ignore
from fmpy.validation import validate_fmu  # type: ignore


class SimpleTable(InputTable):
    """This denotes the concrete table turned into an FMU.
    Exposed interface variables are 'outs' (array) and 'interpolate'"""

    def __init__(self, **kwargs):
        #        print("SimpleTable init", kwargs)
        super().__init__(
            name="TestTable",
            description="my description",
            author="Siegfried Eisinger",
            version="0.1",
            table=((0.0, 1, 2, 3), (1.0, 4, 5, 6), (3.0, 7, 8, 9), (7.0, 8, 7, 6)),
            defaultExperiment={"start_time": 0.0, "stop_time": 10.0, "step_size": 0.1},
            outputNames=("x", "y", "z"),
            interpolate=False,
            **kwargs,
        )


def check_expected(value, expected, feature: str):
    if isinstance(expected, float):
        assert abs(value - expected) < 1e-10, f"Expected the {feature} '{expected}', but found the value {value}"
    else:
        assert value == expected, f"Expected the {feature} '{expected}', but found the value {value}"


def _in_interval(x: float, x0: float, x1: float):
    return x0 <= x <= x1 or x1 <= x <= x0


def _linear(t: float, tt: list, xx: list):
    if t <= tt[-1]:
        return np.interp([t], tt, xx)[0]
    else:
        return xx[-1] + (t - tt[-1]) * (xx[-1] - xx[-2]) / (tt[-1] - tt[-2])


def _to_et(file: str, sub: str = "modelDescription.xml"):
    with ZipFile(file) as zp:
        xml = zp.read(sub)
    return ET.fromstring(xml)


def test_inputtable_class(interpolate=False):
    tbl = InputTable(
        "TestTable",
        "my description",
        table=((0.0, 1, 2, 3), (1.0, 4, 5, 6), (3.0, 7, 8, 9), (7.0, 8, 8, 8)),
        outputNames=("x", "y", "z"),
        interpolate=interpolate,
    )
    assert tbl.interpolate == interpolate, f"Interpolation={interpolate} expected. Found {tbl.interpolate}"
    assert all(
        tbl.times[i] == [0.0, 1.0, 3.0, 7.0][i] for i in range(tbl._rows)
    ), f"Expected time array [0,1,3,7]. Found {tbl.times}"
    for r in range(tbl._rows):
        check_expected(
            all(tbl.outputs[r][c] == [[1, 2, 3], [4, 5, 6], [7, 8, 9], [8, 8, 8]][r][c] for c in range(tbl._cols)),
            True,
            f"Error in expected outputs, row {r}. Found {tbl.outputs[r]}",
        )
    if not interpolate:
        assert all(
            tbl._outs.range[0][c] == (1, 8)[c] for c in range(2)
        ), f"Error in expected range of outputs, row 0. Found {tbl.ranges[0]}"
    assert all(
        tbl.outs[c] == (1, 2, 3)[c] for c in range(tbl._cols)
    ), f"Error in expected outs (row 0). Found {tbl.outs}"
    tbl.setup_experiment(1.0)
    check_expected(tbl.start_time, 1.0, "Start time")
    tbl.enter_initialization_mode()  # iterate to row 0 (start_time)
    tbl.set_values(time=5.0)  # should iterate to row 2 and return interval (3,7)
    for r, time in enumerate(tbl.times):
        tbl.set_values(time)
        assert tbl.times[r] == time, f"Exact time {time}=={tbl.times[r]} at row {r}"
        assert all(tbl.outs[c] == tbl.outputs[r][c] for c in range(tbl._cols)), f"Outs at row {r}"
    for i in range(90):
        time = -1.0 + 0.1 * i
        tbl.set_values(time)
        if tbl.times[0] <= time <= tbl.times[-1]:  # no extrapolation
            if interpolate:
                expected = np.array([np.interp(time, tbl.times, tbl.outputs[:, c]) for c in range(tbl._cols)])
                assert all(tbl.outs[k] == expected[k] for k in range(len(tbl.outs))), f"Got {tbl.outs} != {expected}"
        if not interpolate:
            if time <= tbl.times[0]:
                assert all(
                    tbl.outs[k] == tbl.outputs[0, :][k] for k in range(len(tbl.outs))
                ), f"Got {tbl.outs} != {tbl.outputs[0,:]}"
            elif time >= tbl.times[-1]:
                assert all(
                    tbl.outs[k] == tbl.outputs[-1, :][k] for k in range(len(tbl.outs))
                ), f"Got {tbl.outs} != {tbl.outputs[-1,:]}"
            else:  # inside
                for i, t in enumerate(tbl.times):
                    if t > time:
                        assert all(
                            tbl.outs[k] == tbl.outputs[i - 1, :][k] for k in range(len(tbl.outs))
                        ), f"time {time}, row {i}. Got {tbl.outs} != {tbl.outputs[i-1,:]}"
                        break


def test_make_simpletable(interpolate=False):
    asBuilt = Model.build(__file__)  #'../component_model', ])
    info = fmu_info(asBuilt.name)  # this is a formatted string. Not easy to check
    print(f"Info: {info}")
    et = _to_et(asBuilt.name)
    assert et.attrib["fmiVersion"] == "2.0", "FMI Version"
    # similarly other critical issues of the modelDescription can be checked
    assert et.attrib["variableNamingConvention"] == "structured", "Variable naming convention. => use [i] for arrays"
    #    print(et.attrib)
    val = validate_fmu("SimpleTable.fmu")
    assert not len(val), f"Validation of the modelDescription of {asBuilt.name} was not successful. Errors: {val}"


def test_use_fmu(interpolate=True):
    result = simulate_fmu(
        "SimpleTable.fmu",
        stop_time=10.0,
        step_size=0.1,
        validate=True,
        solver="Euler",
        debug_logging=True,
        logger=print,  # fmi_call_logger=print,
        start_values={"interpolate": interpolate},
    )
    plot_result(result)
    if not interpolate:
        for t, x, y, z in result:
            if t > 7:
                assert x == 8 and y == 7 and z == 6, f"Values for t>7 wrong. Found ({x}, {y}, {z}) at t={t}"
            elif t > 3:
                assert x == 7 and y == 8 and z == 9, f"Values for t>3 wrong. Found ({x}, {y}, {z}) at t={t}"
            elif t > 1:
                assert x == 4 and y == 5 and z == 6, f"Values for t>1 wrong. Found ({x}, {y}, {z}) at t={t}"
            elif t > 0:
                assert x == 1 and y == 2 and z == 3, f"Values for t>0 wrong. Found ({x}, {y}, {z}) at t={t}"
    else:
        _t = (0.0, 1.0, 3.0, 7.0)
        _x = (1, 4, 7, 8)
        _y = (2, 5, 8, 7)
        _z = (3, 6, 9, 6)
        for t, x, y, z in result:
            if t == 0:
                tt = 0  #!! results are retrieved prior to the step, i.e. y_i+1 = f(y_i, ...)
            check_expected(_linear(tt, _t, _x), x, f"Linear interpolated x at t={tt}")
            check_expected(_linear(tt, _t, _y), y, f"Linear interpolated y at t={tt}")
            check_expected(_linear(tt, _t, _z), z, f"Linear interpolated z at t={tt}")
            tt = t


if __name__ == "__main__":
    test_make_simpletable(interpolate=True)
    test_inputtable_class(interpolate=False)
    test_inputtable_class(interpolate=True)
    test_use_fmu(interpolate=True)
