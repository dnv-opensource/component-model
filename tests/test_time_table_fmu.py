import shutil
import xml.etree.ElementTree as ET  # noqa: N817
from pathlib import Path
from typing import Any, Iterable
from zipfile import ZipFile

import numpy as np
import pytest
from fmpy.simulation import simulate_fmu
from fmpy.util import fmu_info, plot_result
from fmpy.validation import validate_fmu
from pythonfmu.enums import Fmi2Causality as Causality
from pythonfmu.enums import Fmi2Variability as Variability

from component_model.model import Model


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


@pytest.fixture(scope="session")
def time_table_fmu():
    return _time_table_fmu()


def _time_table_fmu():
    """Make FMU and return .fmu file with path."""
    build_path = Path.cwd()
    build_path.mkdir(exist_ok=True)
    fmu_path = Model.build(
        script=str(Path(__file__).parent.parent / "examples" / "time_table_fmu.py"),
        project_files=[Path(__file__).parent.parent / "examples" / "time_table.py"],
        dest=build_path,
    )
    return fmu_path


@pytest.fixture(scope="session")
def time_table_system_structure(time_table_fmu: Path):
    return _time_table_system_structure(time_table_fmu)


def _time_table_system_structure(time_table_fmu: Path):
    """Make a structure file and return the path"""
    #     path = make_osp_system_structure(
    #         name="TimeTableStructure",
    #         simulators={"tab": {"source": "TimeTableFMU.fmu", "stepSize": 0.1}},  # , "interpolate": 1}},
    #         version="0.1",
    #         start=0.0,
    #         base_step=0.1,
    #         algorithm="fixedStep",
    #         path=Path.cwd(),
    #     )
    shutil.copy(Path(__file__).parent.parent / "examples" / "TimeTableStructure.xml", Path.cwd())
    return Path.cwd() / "TimeTableStructure.xml"


def test_time_table_fmu():
    from examples.time_table_fmu import TimeTableFMU

    tbl = TimeTableFMU(data=[[0, 0, 0], [1, 1, 1], [2, 2, 4], [3, 3, 9]], header=["exp1", "exp2"], interpolate=1)
    assert tbl._cols == 2
    assert tbl._rows == 4
    assert tbl.header == ("exp1", "exp2")
    assert tbl.interpolate == 1
    arrays_equal(tbl.times, (0, 1, 2, 3))
    arrays_equal(tbl.data[1], [1, 1])
    assert tbl._outs.causality == Causality.output
    assert tbl._outs.variability == Variability.continuous
    arrays_equal(tbl._outs.start, tbl.outs)
    assert tbl._interpolate.causality == Causality.parameter
    assert tbl._interpolate.variability == Variability.fixed
    assert tbl._interpolate.start == (tbl.interpolate,)


def _in_interval(x: float, x0: float, x1: float):
    return x0 <= x <= x1 or x1 <= x <= x0


def _to_et(file: str, sub: str = "modelDescription.xml"):
    with ZipFile(file) as zp:
        xml = zp.read(sub)
    return ET.fromstring(xml)


def test_make_time_table(time_table_fmu: Path):
    info = fmu_info(str(time_table_fmu))  # this is a formatted string. Not easy to check
    print(f"Info: {info}")
    et = _to_et(str(time_table_fmu))
    assert et.attrib["fmiVersion"] == "2.0", "FMI Version"
    # similarly other critical issues of the modelDescription can be checked
    assert et.attrib["variableNamingConvention"] == "structured", "Variable naming convention. => use [i] for arrays"
    #    print(et.attrib)
    val = validate_fmu(str(time_table_fmu))
    assert not len(val), (
        f"Validation of the modelDescription of {time_table_fmu.name} was not successful. Errors: {val}"
    )


def test_use_fmu(time_table_fmu: Path, show: bool = False):
    """Use the FMU running it on fmpy using various interpolate settings."""
    print(fmu_info(str(time_table_fmu)))
    _t = np.linspace(0, 10, 101)
    for ipol in range(4):
        result = simulate_fmu(
            time_table_fmu,
            stop_time=10.0,
            step_size=0.1,
            validate=True,
            solver="Euler",
            debug_logging=True,
            logger=print,  # fmi_call_logger=print,
            start_values={"interpolate": ipol},
        )
        if show:
            plot_result(result)
        for i, t in enumerate(_t):
            assert result[i][0] == t, f"Wrong time picked: {t} != {result[i][0]}"
            if ipol >= 0:
                pass
            elif ipol == 0:
                assert result[i][1] == 1.0, f"Result for {ipol}, time={t}: {result[i][1]} != 1.0"
            elif ipol == 1:
                assert abs(result[i][2] - t) < 1e-10, f"Result for {ipol}, time={t}: {result[i][1]} != {i}"
            elif ipol == 2:
                assert abs(result[i][3] - t**2) < 1e-10, f"Result for {ipol}, time={t}: {result[i][1]} != {i**2}"


def test_make_with_new_data():
    """Test and example how keyword arguments of the Model class can be used (changed) when building FMU."""
    times = np.linspace(0, 2 * np.pi, 100)
    data = list(zip(times, np.cos(times), np.sin(times), strict=False))
    build_path = Path.cwd()
    build_path.mkdir(exist_ok=True)
    _ = Model.build(
        script=str(Path(__file__).parent.parent / "examples" / "time_table_fmu.py"),
        project_files=[Path(__file__).parent.parent / "examples" / "time_table.py"],
        dest=build_path,
        newargs={
            "data": data,
            "header": ("x", "y"),
            "interpolate": 1,
            "default_experiment": {"startTime": 0, "stopTime": 2 * np.pi, "stepSize": 0.1, "tolerance": 1e-5},
        },
    )


# @pytest.mark.skip(reason="Does so far not work within pytest, only stand-alone")
def test_use_with_new_data(show: bool):
    fmu_path = Path(__file__).parent / "test_working_directory" / "TimeTableFMU.fmu"
    result = simulate_fmu(
        fmu_path,
        stop_time=2 * np.pi,
        step_size=0.1,
        validate=True,
        solver="Euler",
        debug_logging=True,
        logger=print,  # fmi_call_logger=print,
        start_values={"interpolate": 2},
    )
    if show:
        plot_result(result)
    time = 0.0
    for t, x, y in result:
        assert abs(t - time) < 1e-10, f"Expected time {time}!={t} from data"
        assert abs(1.0 - (x**2 + y**2)) < 1e-5, f"time {t}: x={x}, y={y}, x**2+y**2 = {x**2 + y**2}"
        time += 0.1
        if time > 2 * np.pi:
            time = 2 * np.pi


if __name__ == "__main__":
    retcode = pytest.main(["-rA", "-v", __file__])
    assert retcode == 0, f"Non-zero return code {retcode}"
    import os

    os.chdir(Path(__file__).parent.absolute() / "test_working_directory")
    # test_time_table_fmu()
    # test_make_time_table(_time_table_fmu())
    # test_use_fmu(_time_table_fmu(), show=True)
    # test_make_with_new_data()
    # test_use_with_new_data(show=True)
