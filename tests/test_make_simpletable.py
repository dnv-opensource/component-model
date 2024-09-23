import os
import xml.etree.ElementTree as ET  # noqa: N817
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest
from component_model.example_models.input_table import InputTable  # type: ignore
from component_model.model import Model  # type: ignore
from fmpy import simulate_fmu  # type: ignore
from fmpy.util import fmu_info  # type: ignore
from fmpy.validation import validate_fmu  # type: ignore
from libcosimpy.CosimEnums import CosimExecutionState
from libcosimpy.CosimExecution import CosimExecution
from libcosimpy.CosimManipulator import CosimManipulator  # type: ignore
from libcosimpy.CosimObserver import CosimObserver  # type: ignore
from libcosimpy.CosimSlave import CosimLocalSlave


def check_expected(value, expected, feature: str):
    if isinstance(expected, float):
        assert abs(value - expected) < 1e-10, f"Expected the {feature} '{expected}', but found the value {value}"
    else:
        assert value == expected, f"Expected the {feature} '{expected}', but found the value {value}"


@pytest.fixture(scope="session")
def simple_table_fmu():
    build_path = Path.cwd() / "fmus"
    build_path.mkdir(exist_ok=True)
    fmu_path = Model.build(
        str(Path(__file__).parent / "resources" / "simple_table.py"),
        project_files=[],
        dest=build_path,
    )
    return fmu_path


@pytest.fixture(scope="session")
def simple_table_system_structure(tmp_path_factory, simple_table_fmu):
    ET.register_namespace("", "http://opensimulationplatform.com/MSMI/OSPSystemStructure")
    tree = ET.parse(Path(__file__).parent / "resources" / "SimpleTableSystemStructure.xml")
    root = tree.getroot()

    root[0][0].attrib["source"] = f"../{os.path.basename(simple_table_fmu.parent)}/SimpleTable.fmu"

    build_path = Path.cwd() / "config"
    build_path.mkdir(exist_ok=True)
    system_structure_path = build_path / "SimpleTableSystemStructure.xml"
    tree.write(system_structure_path)
    return system_structure_path


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
            tbl._outs.range[0][c] == (float("-inf"), float("inf"))[c] for c in range(2)
        ), f"Error in expected range of outputs, row 0. Found {tbl._outs.range[0]}"
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


def test_make_simpletable(simple_table_fmu):
    info = fmu_info(simple_table_fmu)  # this is a formatted string. Not easy to check
    print(f"Info: {info}")
    et = _to_et(str(simple_table_fmu))
    assert et.attrib["fmiVersion"] == "2.0", "FMI Version"
    # similarly other critical issues of the modelDescription can be checked
    assert et.attrib["variableNamingConvention"] == "structured", "Variable naming convention. => use [i] for arrays"
    #    print(et.attrib)
    val = validate_fmu(str(simple_table_fmu))
    assert not len(
        val
    ), f"Validation of the modelDescription of {simple_table_fmu.name} was not successful. Errors: {val}"


def test_use_fmu_interpolation(simple_table_fmu):
    result = simulate_fmu(
        simple_table_fmu,
        stop_time=10.0,
        step_size=0.1,
        validate=True,
        solver="Euler",
        debug_logging=True,
        logger=print,  # fmi_call_logger=print,
        start_values={"interpolate": True},
    )
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


def test_use_fmu_no_interpolation(simple_table_fmu):
    result = simulate_fmu(
        simple_table_fmu,
        stop_time=10.0,
        step_size=0.1,
        validate=True,
        solver="Euler",
        debug_logging=True,
        logger=print,  # fmi_call_logger=print,
        start_values={"interpolate": False},
    )

    for t, x, y, z in result:
        if t > 7:
            assert x == 8 and y == 7 and z == 6, f"Values for t>7 wrong. Found ({x}, {y}, {z}) at t={t}"
        elif t > 3:
            assert x == 7 and y == 8 and z == 9, f"Values for t>3 wrong. Found ({x}, {y}, {z}) at t={t}"
        elif t > 1:
            assert x == 4 and y == 5 and z == 6, f"Values for t>1 wrong. Found ({x}, {y}, {z}) at t={t}"
        elif t > 0:
            assert x == 1 and y == 2 and z == 3, f"Values for t>0 wrong. Found ({x}, {y}, {z}) at t={t}"


def test_run_osp(simple_table_fmu):
    sim = CosimExecution.from_step_size(step_size=1e8)  # empty execution object with fixed time step in nanos
    st = CosimLocalSlave(fmu_path=str(simple_table_fmu), instance_name="st")

    ist = sim.add_local_slave(st)
    assert ist == 0, f"local slave number {ist}"

    reference_dict = {var_ref.name.decode(): var_ref.reference for var_ref in sim.slave_variables(ist)}

    # Set initial values
    sim.boolean_initial_value(ist, reference_dict["interpolate"], True)

    sim_status = sim.status()
    assert sim_status.current_time == 0
    assert CosimExecutionState(sim_status.state) == CosimExecutionState.STOPPED

    # Simulate for 1 second
    sim.simulate_until(target_time=15e9)


def test_run_osp_system_structure(simple_table_system_structure):
    "Run an OSP simulation in the same way as the SimulatorInterface of case_study is implemented"
    simulator = CosimExecution.from_osp_config_file(str(simple_table_system_structure))
    comps = []
    for comp in list(simulator.slave_infos()):
        name = comp.name.decode()
        comps.append(name)
    assert comps == ["tab"], f"Components: {comps}"
    variables = {}
    for idx in range(simulator.num_slave_variables(0)):
        struct = simulator.slave_variables(0)[idx]
        variables.update(
            {
                struct.name.decode(): {
                    "reference": struct.reference,
                    "type": struct.type,
                    "causality": struct.causality,
                    "variability": struct.variability,
                }
            }
        )
    assert variables["outs[0]"] == {"reference": 0, "type": 0, "causality": 2, "variability": 4}  # similar: [1],[2]
    assert variables["interpolate"] == {"reference": 3, "type": 3, "causality": 1, "variability": 1}

    # Instantiate a suitable manipulator for changing variables.
    manipulator = CosimManipulator.create_override()
    simulator.add_manipulator(manipulator=manipulator)
    simulator.boolean_initial_value(0, 3, True)
    # Instantiate a suitable observer for collecting results.
    observer = CosimObserver.create_last_value()
    simulator.add_observer(observer=observer)
    for time in range(1, 10):
        simulator.simulate_until(time * 1e9)
        values = observer.last_real_values(0, [0, 1, 2])
        print(f"Time {time/1e9}: {values}")
        if time == 5:
            assert values == [7.475, 7.525, 7.574999999999999]
