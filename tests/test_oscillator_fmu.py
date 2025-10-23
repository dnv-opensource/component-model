import shutil
import sys
import xml.etree.ElementTree as ET  # noqa: N817
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytest
from fmpy.simulation import simulate_fmu
from fmpy.util import fmu_info, plot_result
from fmpy.validation import validate_fmu
from libcosimpy.CosimEnums import (
    CosimExecutionState,
    CosimVariableCausality,
    CosimVariableType,
    CosimVariableVariability,
)
from libcosimpy.CosimExecution import CosimExecution
from libcosimpy.CosimLogging import CosimLogLevel, log_output_level
from libcosimpy.CosimManipulator import CosimManipulator
from libcosimpy.CosimObserver import CosimObserver
from libcosimpy.CosimSlave import CosimLocalSlave

from component_model.model import Model
from component_model.utils.xml import read_xml


def _slave_index(simulator: CosimExecution, name: str):
    """Get the slave index from the name."""
    return simulator.slave_index_from_instance_name(name)


def _var_ref(simulator: CosimExecution, slave_name: str, var_name: str):
    """Get the variable value reference from slave and variable name.
    Return both the slave index and the value reference as tuple."""
    slave = _slave_index(simulator, slave_name)
    if slave is None:
        return (-1, -1)
    for idx in range(simulator.num_slave_variables(slave)):
        struct = simulator.slave_variables(slave)[idx]
        if struct.name.decode() == var_name:
            return (slave, struct.reference)
    return (-1, -1)


def _check_var(simulator: CosimExecution, slave_name: str, var_name: str, expected: int | tuple | None):
    if var_name == "":
        idx = _slave_index(simulator, slave_name)
        assert idx == expected, f"Slave {slave_name}. Index: {idx}. Expected: {expected}"
    else:
        idx, vref = _var_ref(simulator, slave_name, var_name)
        assert (idx, vref) == expected, f"Variable {slave_name}.{var_name}: ({idx},{vref}). Expected:{expected}"


def _var_info(simulator: CosimExecution, slave_name: str, var_name: str):
    slave = _slave_index(simulator, slave_name)
    if slave is None:
        return {}
    for idx in range(simulator.num_slave_variables(slave)):
        struct = simulator.slave_variables(slave)[idx]
        if struct.name.decode() == var_name:
            return {
                "slave": 0,
                "idx": idx,
                "reference": struct.reference,
                "type": CosimVariableType(struct.type).name,
                "causality": CosimVariableCausality(struct.causality).name,
                "variability": CosimVariableVariability(struct.variability).name,
            }
    return {}


def _var_list(simulator: CosimExecution, slave_name: str, ret: str = "print"):
    slave = _slave_index(simulator, slave_name)
    assert slave is not None, f"Slave {slave_name} not found in system"
    variables = {}
    for idx in range(simulator.num_slave_variables(slave)):
        struct = simulator.slave_variables(slave)[idx]
        if ret == "print":
            print(
                f"Slave {slave_name}({slave}), var {struct.name.decode()}({struct.reference}), type: {CosimVariableType(struct.type).name}, causality: {CosimVariableCausality(struct.causality).name}, variability: {CosimVariableVariability(struct.variability)}"
            )
        else:
            variables[struct.name.decode()] = {
                "slave": slave,
                "idx": idx,
                "reference": struct.reference,
                "type": CosimVariableType(struct.type).name,
                "causality": CosimVariableCausality(struct.causality).name,
                "variability": CosimVariableVariability(struct.variability),
            }


def system_structure_change(structure_file: Path, tag: str, what: str, newval: str):
    def register_all_namespaces(filename):
        namespaces: dict = {}
        for _, (ns, uri) in ET.iterparse(filename, events=["start-ns"]):
            # print("TYPES", ns, type(ns), uri, type(uri))
            namespaces.update({ns: uri})
            ET.register_namespace(str(ns), str(uri))
        #         namespaces: dict = dict([node )])
        #        for ns in namespaces:
        #            ET.register_namespace(ns, namespaces[ns])
        return namespaces

    nss = register_all_namespaces(structure_file)
    el = read_xml(structure_file)
    elements = el.findall(f"ns:{tag}", {"ns": nss[""]})
    for e in elements:
        if what == "text":
            e.text = newval
        else:  # assume attribute name
            e.attrib[what] = newval
    ET.ElementTree(el).write(structure_file.name, encoding="utf-8")


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


def do_show(traces: dict[str, tuple[list[float], list[float]]]):
    fig, ax = plt.subplots()
    for label, trace in traces.items():
        _ = ax.plot(trace[0], trace[1], label=label)
    _ = ax.legend()
    plt.show()


# def force(t: float, ampl: float = 1.0, omega: float = 0.1):
#     return np.array((0, 0, ampl * np.sin(omega * t)), dtype=float)


@pytest.fixture(scope="session")
def oscillator_fmu():
    return _oscillator_fmu()


def _oscillator_fmu():
    """Make FMU and return .fmu file with path."""
    build_path = Path.cwd()
    build_path.mkdir(exist_ok=True)
    src = Path(__file__).parent.parent / "examples" / "oscillator_fmu.py"
    fmu_path = Model.build(
        script=src,
        dest=build_path,
    )
    return fmu_path


@pytest.fixture(scope="session")
def driver_fmu():
    return _driver_fmu()


def _driver_fmu():
    """Make FMU and return .fmu file with path."""
    build_path = Path.cwd()
    build_path.mkdir(exist_ok=True)
    src = Path(__file__).parent.parent / "examples" / "driving_force_fmu.py"
    fmu_path = Model.build(
        script=src,
        dest=build_path,
        newargs={"ampl": ("3N", "2N", "1N"), "freq": ("3Hz", "2Hz", "1Hz")},
    )
    return fmu_path


@pytest.fixture(scope="session")
def system_structure():
    return _system_structure()


def _system_structure():
    """Make a OSP structure file and return the path"""
    shutil.copy(Path(__file__).parent.parent / "examples" / "ForcedOscillator.xml", Path.cwd())
    return Path.cwd() / "ForcedOscillator.xml"


def test_make_fmus(
    oscillator_fmu: Path,
    driver_fmu: Path,
):
    info = fmu_info(filename=str(oscillator_fmu))  # this is a formatted string. Not easy to check
    print(f"Info Oscillator: {info}")
    val = validate_fmu(filename=str(oscillator_fmu))
    assert not len(val), f"Validation of of {oscillator_fmu.name} was not successful. Errors: {val}"

    info = fmu_info(filename=str(driver_fmu))  # this is a formatted string. Not easy to check
    print(f"Info Driver: {info}")
    val = validate_fmu(filename=str(driver_fmu))
    assert not len(val), f"Validation of of {driver_fmu.name} was not successful. Errors: {val}"


# def test_make_system_structure(system_structure: Path):
#     assert Path(system_structure).exists(), "System structure not created"
#     el = read_xml(Path(system_structure))
#     assert isinstance(el, ET.Element), f"ElementTree element expected. Found {el}"
#     ns = el.tag.split("{")[1].split("}")[0]
#     print("NS", ns, system_structure)
#     for s in el.findall(".//{*}Simulator"):
#         assert (Path(system_structure).parent / s.get("source", "??")).exists(), f"Component {s.get('name')} not found"
#     for _con in el.findall(".//{*}VariableConnection"):
#         for c in _con:
#             assert c.attrib in ({"simulator": "drv", "name": "f[2]"}, {"simulator": "osc", "name": "f[2]"})
#


def test_run_fmpy(oscillator_fmu: Path, driver_fmu: Path, show: bool = True):
    """Test single FMUs."""
    # sourcery skip: move-assign
    result = simulate_fmu(
        oscillator_fmu,
        stop_time=50,
        step_size=0.01,
        validate=True,
        solver="Euler",
        debug_logging=True,
        logger=print,  # fmi_call_logger=print,
        start_values={"x[2]": 1.0, "c[2]": 0.1},
        step_finished=None,  # pyright: ignore[reportArgumentType]  # (typing incorrect in fmpy)
        fmu_instance=None,  # pyright: ignore[reportArgumentType]  # (typing incorrect in fmpy)
    )
    if show:
        plot_result(result)


def test_run_osp(oscillator_fmu: Path, driver_fmu: Path):
    # sourcery skip: extract-duplicate-method
    sim = CosimExecution.from_step_size(step_size=1e8)  # empty execution object with fixed time step in nanos
    osc = CosimLocalSlave(fmu_path=str(oscillator_fmu), instance_name="osc")
    _osc = sim.add_local_slave(osc)
    assert _osc == 0, f"local slave number {_osc}"

    dri = CosimLocalSlave(fmu_path=str(driver_fmu), instance_name="dri")
    _dri = sim.add_local_slave(dri)
    assert _dri == 1, f"local slave number {_dri}"

    _check_var(sim, "dri", "", _dri)
    _check_var(sim, "dri", "d_freq[2]", (1, 8))
    _check_var(sim, "dri", "f[2]", (1, 11))
    info = _var_info(sim, "dri", "f[2]")
    assert info["causality"] == "OUTPUT"
    assert info["variability"] == "CONTINUOUS"
    assert info["type"] == "REAL"

    # Set initial values
    sim.real_initial_value(*_var_ref(sim, "osc", "x[2]"), value=1.0)
    sim.real_initial_value(*_var_ref(sim, "osc", "c[2]"), value=0.1)

    sim_status = sim.status()
    assert sim_status.current_time == 0
    assert CosimExecutionState(sim_status.state) == CosimExecutionState.STOPPED

    # Simulate for 1 second
    _ = sim.simulate_until(target_time=15e9)


@pytest.mark.skipif(sys.platform.startswith("linux"), reason="HarmonicOsciallatorFMU.fmu throws an error on Linux")
def test_run_osp_system_structure(system_structure: Path, show: bool = True):
    "Run an OSP simulation in the same way as the SimulatorInterface of sim-explorer is implemented"
    log_output_level(CosimLogLevel.TRACE)
    print("STRUCTURE", system_structure)
    try:
        sim = CosimExecution.from_osp_config_file(str(system_structure))
    except Exception as err:
        print("ERR", err)
        return
    sim_status = sim.status()
    assert sim_status.current_time == 0
    assert CosimExecutionState(sim_status.state).name == "STOPPED"
    _check_var(sim, "osc", "", 0)
    _check_var(sim, "drv", "", 1)
    # _var_list(sim, 'drv')
    _check_var(sim, "drv", "ampl[0]", (1, 0))
    _check_var(sim, "drv", "ampl[1]", (1, 1))
    _check_var(sim, "drv", "ampl[2]", (1, 2))
    _check_var(sim, "drv", "d_freq[2]", (1, 8))
    _check_var(sim, "drv", "f[2]", (1, 11))
    info = _var_info(sim, "drv", "f[2]")
    assert info["causality"] == "OUTPUT"
    assert info["variability"] == "CONTINUOUS"
    assert info["type"] == "REAL"

    info = _var_info(sim, "osc", "c[2]")
    assert info["type"] == "REAL"
    assert info["causality"] == "PARAMETER"
    assert info["variability"] == "FIXED"

    info = _var_info(sim, "osc", "x[2]")
    assert info["type"] == "REAL"
    assert info["causality"] == "OUTPUT"
    assert info["variability"] == "CONTINUOUS"

    info = _var_info(sim, "osc", "v[2]")
    assert info["type"] == "REAL"
    assert info["causality"] == "OUTPUT"
    assert info["variability"] == "CONTINUOUS"

    # Instantiate a suitable observer for collecting results.
    # Instantiate a suitable manipulator for changing variables.
    manipulator = CosimManipulator.create_override()
    sim.add_manipulator(manipulator=manipulator)
    sim.real_initial_value(*_var_ref(sim, "osc", "c[2]"), value=0.5)  # c[2]
    sim.real_initial_value(*_var_ref(sim, "osc", "x[2]"), value=1.0)  # x[2]
    observer = CosimObserver.create_last_value()
    sim.add_observer(observer=observer)
    times: list[float] = []
    pos: list[float] = []
    speed: list[float] = []
    slave, x2_ref = _var_ref(sim, "osc", "x[2]")
    slave, v2_ref = _var_ref(sim, "osc", "v[2]")
    for step in range(1, 1000):
        time = step * 0.01
        _ = sim.simulate_until(step * 1e8)
        values = observer.last_real_values(slave_index=0, variable_references=[x2_ref, v2_ref])
        times.append(time)
        pos.append(values[0])
        speed.append(values[1])
    if show:
        do_show(traces={"z-pos": (times, pos), "z-speed": (times, speed)})


def test_system_structure_change(system_structure):
    system_structure_change(system_structure, "BaseStepSize", "text", str(0.99))


@pytest.mark.parametrize("alg", ["fixedStep", "ecco"])
def test_run_osp_sweep(system_structure: Path, alg: str, show: bool = True):
    _test_run_osp_sweep(system_structure, show, alg)


def _test_run_osp_sweep(system_structure: Path, show: bool = True, alg: str = "fixedStep"):
    "Run an OSP simulation of the oscillator and the force sweep as co-simulation."

    dt = 1.0
    t_end = 100.0

    log_output_level(CosimLogLevel.TRACE)
    system_structure_change(system_structure, "BaseStepSize", "text", str(dt))
    system_structure_change(system_structure, "Algorithm", "text", alg)
    print(f"Running Algorithm {alg} on {system_structure}")
    assert system_structure.exists(), f"File {system_structure} not found"
    sim = CosimExecution.from_osp_config_file(str(system_structure))
    sim_status = sim.status()
    assert sim_status.error_code == 0
    # manipulator = CosimManipulator.create_override()
    # sim.add_manipulator(manipulator=manipulator)
    sim.real_initial_value(*_var_ref(sim, "osc", "k[2]"), value=1.0)
    sim.real_initial_value(*_var_ref(sim, "osc", "c[2]"), value=0.1)
    sim.real_initial_value(*_var_ref(sim, "osc", "m"), value=1.0)
    sim.real_initial_value(*_var_ref(sim, "drv", "ampl[2]"), value=1.0)
    sim.real_initial_value(*_var_ref(sim, "drv", "freq[2]"), value=0.0)  # start frequency
    sim.real_initial_value(*_var_ref(sim, "drv", "d_freq[2]"), value=0.1 / 2 / np.pi)
    sim.real_initial_value(*_var_ref(sim, "osc", "x[2]"), value=0.0)
    sim.real_initial_value(*_var_ref(sim, "osc", "v[2]"), value=0.0)
    observer = CosimObserver.create_last_value()
    sim.add_observer(observer=observer)
    times = []
    pos = []
    speed = []
    time = 0.0
    slave, x2_ref = _var_ref(sim, "osc", "x[2]")
    slave, v2_ref = _var_ref(sim, "osc", "v[2]")
    while time < t_end:
        time += dt
        times.append(time)
        _ = sim.simulate_until(int(time * 1e9))
        values = observer.last_real_values(slave_index=0, variable_references=[x2_ref, v2_ref])
        pos.append(values[0])
        speed.append(values[1])
    if Path("oscillator_sweep0.dat").exists():
        times0, pos0, speed0, force0 = [], [], [], []
        with open("oscillator_sweep0.dat", "r") as fp:
            for line in fp:
                t, p, v, f = line.split("\t")
                times0.append(float(t))
                pos0.append(float(p))
                speed0.append(float(v))
                force0.append(float(f))
        if show:
            freq0 = [0.1 * t / 2 / np.pi for t in times0]
            freq = [0.1 * t / 2 / np.pi for t in times]
            do_show({"z-pos0": (freq0, pos0), "z-pos": (freq, pos)})

    elif show:
        do_show({"z-pos": (times, pos), "z-speed": (times, speed)})


if __name__ == "__main__":
    retcode = pytest.main(args=["-rA", "-v", __file__, "--show", "True"])
    assert retcode == 0, f"Non-zero return code {retcode}"
    import os

    os.chdir(Path(__file__).parent.absolute() / "test_working_directory")
    # test_system_structure_change(_system_structure())
    # test_make_fmus(_oscillator_fmu(), _driver_fmu())
    # test_run_fmpy(_oscillator_fmu(), _driver_fmu(), show=True)
    # test_run_osp(_oscillator_fmu(), _driver_fmu())
    # test_run_osp_system_structure(_system_structure(), show=True)
    # _test_run_osp_sweep(_system_structure(), show=True, alg="fixedStep")
    # _test_run_osp_sweep( _system_structure(), show=True, alg='ecco')
