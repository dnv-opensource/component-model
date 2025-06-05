import shutil
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytest
from fmpy.simulation import simulate_fmu
from fmpy.util import fmu_info, plot_result
from fmpy.validation import validate_fmu
from libcosimpy.CosimEnums import CosimExecutionState
from libcosimpy.CosimExecution import CosimExecution
from libcosimpy.CosimLogging import CosimLogLevel, log_output_level
from libcosimpy.CosimManipulator import CosimManipulator
from libcosimpy.CosimObserver import CosimObserver
from libcosimpy.CosimSlave import CosimLocalSlave

from component_model.model import Model
from component_model.utils.xml import read_xml
import xml.etree.ElementTree as ET  # noqa: N817

def system_structure_change( structure_file: Path, tag: str, what: str, newval: str):
    def register_all_namespaces(filename):
        namespaces = dict([node for _, node in ET.iterparse(filename, events=['start-ns'])])
        for ns in namespaces:
            ET.register_namespace(ns, namespaces[ns])
        return namespaces

    nss = register_all_namespaces( structure_file)
    el = read_xml(structure_file)
    elements = el.findall(f"ns:{tag}", {'ns' : nss['']})
    for e in elements:
        if what=='text':
            e.text = newval
        else: # assume attribute name
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

def do_show( traces: dict[str, tuple[list[float],list[float]]]):
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
        script=str(src),
        project_files=[src],
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
        script=str(src),
        project_files=[src],
        dest=build_path
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


def test_use_fmu(oscillator_fmu: Path, driver_fmu: Path, show: bool):
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
    reference_dict = {var_ref.name.decode(): var_ref.reference for var_ref in sim.slave_variables(_osc)}

    dri = CosimLocalSlave(fmu_path=str(driver_fmu), instance_name="dri")
    _dri = sim.add_local_slave(dri)
    assert _dri == 1, f"local slave number {_dri}"

    # Set initial values
    sim.real_initial_value(slave_index=_osc, variable_reference=reference_dict["x[2]"], value=1.0)
    sim.real_initial_value(slave_index=_osc, variable_reference=reference_dict["c[2]"], value=0.1)

    sim_status = sim.status()
    assert sim_status.current_time == 0
    assert CosimExecutionState(sim_status.state) == CosimExecutionState.STOPPED

    # Simulate for 1 second
    _ = sim.simulate_until(target_time=15e9)


@pytest.mark.skipif(sys.platform.startswith("linux"), reason="HarmonicOsciallatorFMU.fmu throws an error on Linux")
def test_run_osp_system_structure(system_structure: Path, show: bool):
    "Run an OSP simulation in the same way as the SimulatorInterface of sim-explorer is implemented"
    log_output_level(CosimLogLevel.TRACE)
    print("STRUCTURE", system_structure)
    try:
        simulator = CosimExecution.from_osp_config_file(str(system_structure))
    except Exception as err:
        print("ERR", err)
    sim_status = simulator.status()
    assert sim_status.current_time == 0
    assert CosimExecutionState(sim_status.state) == CosimExecutionState.STOPPED
    comps = []
    for comp in list(simulator.slave_infos()):
        name = comp.name.decode()
        comps.append(name)
    assert comps == ["osc", "drv"]
    variables = {}
    for idx in range(simulator.num_slave_variables(0)):
        struct = simulator.slave_variables(0)[idx]
        variables[struct.name.decode()] = {
            "reference": struct.reference,
            "type": struct.type,
            "causality": struct.causality,
            "variability": struct.variability,
        }

    for idx in range(simulator.num_slave_variables(1)):
        struct = simulator.slave_variables(1)[idx]
        variables |= {
            struct.name.decode(): {
                "reference": struct.reference,
                "type": struct.type,
                "causality": struct.causality,
                "variability": struct.variability,
            }
        }
    assert variables["c[2]"]["type"] == 0
    assert variables["c[2]"]["causality"] == 1
    assert variables["c[2]"]["variability"] == 1

    assert variables["x[2]"]["type"] == 0
    assert variables["x[2]"]["causality"] == 2
    assert variables["x[2]"]["variability"] == 4

    assert variables["v[2]"]["type"] == 0
    assert variables["v[2]"]["causality"] == 2
    assert variables["v[2]"]["variability"] == 4

    # Instantiate a suitable observer for collecting results.
    # Instantiate a suitable manipulator for changing variables.
    manipulator = CosimManipulator.create_override()
    simulator.add_manipulator(manipulator=manipulator)
    simulator.real_initial_value(slave_index=0, variable_reference=5, value=0.5)  # c[2]
    simulator.real_initial_value(slave_index=0, variable_reference=9, value=1.0)  # x[2]
    observer = CosimObserver.create_last_value()
    simulator.add_observer(observer=observer)
    times = []
    pos = []
    speed = []
    for step in range(1, 1000):
        time = step * 0.01
        _ = simulator.simulate_until(step * 1e8)
        values = observer.last_real_values(slave_index=0, variable_references=[9, 12])  # x[2], v[2]
        times.append(time)
        pos.append(values[0])
        speed.append(values[1])
    if show:
        do_show(traces={'z-pos' : (times,pos), 'z-speed' : (times,speed)})

@pytest.mark.parameterize("alg", ['fixedStep', 'ecco'])
def test_run_osp_sweep(system_structure: Path, show: bool, alg:str):
    "Run an OSP simulation of the oscillator and the force sweep as co-simulation."

    dt = 1.0
    t_end = 100.0
    
    log_output_level(CosimLogLevel.TRACE)
    system_structure_change(system_structure, "BaseStepSize", "text", str(dt))
    system_structure_change(system_structure, "Algorithm", "text", alg)
    print(f"Running Algorithm {alg} on {system_structure}")
    assert system_structure.exists(), f"File {system_structure} not found"
    simulator = CosimExecution.from_osp_config_file(str(system_structure))
    print(get_last_error_message())
    sim_status = simulator.status()
    #manipulator = CosimManipulator.create_override()
    #simulator.add_manipulator(manipulator=manipulator)
    simulator.real_initial_value(slave_index=0, variable_reference=2, value=1.0)           # k[2]
    simulator.real_initial_value(slave_index=0, variable_reference=5, value=0.1)           # c[2]
    simulator.real_initial_value(slave_index=0, variable_reference=6, value=1.0)           # m
    simulator.real_initial_value(slave_index=0, variable_reference=0, value=1.0)           # ampl
    simulator.real_initial_value(slave_index=1, variable_reference=1, value=0.0)           # freq (start frequency)
    simulator.real_initial_value(slave_index=1, variable_reference=2, value=0.1/2/np.pi)  # d_freq 
    simulator.real_initial_value(slave_index=0, variable_reference=9, value=0.0)           # x0[2]
    simulator.real_initial_value(slave_index=0, variable_reference=12, value=0.0)          # v0[2]
    observer = CosimObserver.create_last_value()
    simulator.add_observer(observer=observer)
    times = []
    pos = []
    speed = []
    time = 0.0
    while time < t_end:
        time += dt
        times.append( time)
        _ = simulator.simulate_until( int(time*1e9))
        values = observer.last_real_values(slave_index=0, variable_references=[9, 12])  # x[2], v[2]
        pos.append(values[0])
        speed.append(values[1])
    if Path("oscillator_sweep0.dat").exists():
        times0, pos0, speed0, force0 = [], [], [], []
        with open("oscillator_sweep0.dat", 'r') as fp:
            for line in fp:
                t, p, v, f = line.split('\t')
                times0.append(float(t))
                pos0.append(float(p))
                speed0.append(float(v))
                force0.append(float(f))
        if show:
            freq0 = [0.1*t/2/np.pi for t in times0]
            freq = [0.1*t/2/np.pi for t in times]
            do_show({'z-pos0' : (freq0,pos0), 'z-pos' : (freq,pos)})
            
    elif show:
        do_show( {'z-pos' : (times,pos), 'z-speed' : (times,speed)})
    

if __name__ == "__main__":
    retcode = 0 #pytest.main(args=["-rA", "-v", __file__, "--show", "True"])
    assert retcode == 0, f"Non-zero return code {retcode}"
    import os

    os.chdir(Path(__file__).parent.absolute() / "test_working_directory")
    test_make_fmus(_oscillator_fmu(), _driver_fmu())
    # test_make_system_structure( _system_structure())
    # test_use_fmu(_oscillator_fmu(), _driver_fmu(), show=True)
    # test_run_osp(_oscillator_fmu(), _driver_fmu())
    # test_run_osp_system_structure(_system_structure(), show=True)
    # test_run_osp_sweep( _system_structure(), show=True, alg='fixedStep')
    test_run_osp_sweep( _system_structure(), show=True, alg='ecco')
