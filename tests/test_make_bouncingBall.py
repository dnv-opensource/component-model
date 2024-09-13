import xml.etree.ElementTree as ET  # noqa: N817
from zipfile import ZipFile

import matplotlib.pyplot as plt
import pytest
from component_model.example_models.bouncing_ball import BouncingBall  # type: ignore
from component_model.model import Model  # type: ignore
from fmpy import plot_result, simulate_fmu  # type: ignore
from fmpy.util import fmu_info  # type: ignore
from fmpy.validation import validate_fmu  # type: ignore
from libcosimpy.CosimEnums import CosimExecutionState
from libcosimpy.CosimExecution import CosimExecution
from libcosimpy.CosimSlave import CosimLocalSlave


def _in_interval(x: float, x0: float, x1: float):
    return x0 <= x <= x1 or x1 <= x <= x0


def arrays_equal(arr1, arr2, dtype="float64", eps=1e-7):
    assert len(arr1) == len(arr2), "Length not equal!"

    for i in range(len(arr1)):
        # assert type(arr1[i]) == type(arr2[i]), f"Array element {i} type {type(arr1[i])} != {type(arr2[i])}"
        assert abs(arr1[i] - arr2[i]) < eps, f"Component {i}: {arr1[i]} != {arr2[i]}"


def _to_et(file: str, sub: str = "modelDescription.xml"):
    with ZipFile(file) as zp:
        xml = zp.read(sub)
    return ET.fromstring(xml)


def test_bouncing_ball_class():
    bb = BouncingBall(pos=(0, 0, 10), speed=(1, 0, 0), g=9.81, e=0.9, min_speed_z=1e-6)
    arrays_equal(bb.pos, (0, 0, 0.254))  # 1)) was provided as inch
    assert bb.g == 9.81
    time = 0
    dt = bb.default_experiment["stepSize"]
    height = [bb.pos[2]]
    times = [0]
    for _ in range(200):
        bb.do_step(time, dt)
        time += dt
        height.append(bb.pos[2])
        times.append(time)
    # print( height)
    plt.plot(times, height)
    plt.show()


def test_make_bouncing_ball():
    asBuilt = Model.build("../component_model/example_models/bouncing_ball.py", project_files=[])
    info = fmu_info(asBuilt.name)  # not necessary, but it lists essential properties of the FMU
    print(f"Info: {info}")
    et = _to_et(asBuilt.name)
    assert et.attrib["fmiVersion"] == "2.0", "FMI Version"
    # similarly other critical issues of the modelDescription can be checked
    assert et.attrib["variableNamingConvention"] == "structured", "Variable naming convention. => use [i] for arrays"
    #    print(et.attrib)
    val = validate_fmu("BouncingBall.fmu")
    assert not len(val), f"Validation of the modelDescription of {asBuilt.name} was not successful. Errors: {val}"


def test_use_fmu():
    result = simulate_fmu(
        "BouncingBall.fmu",
        stop_time=3.0,
        step_size=0.1,
        validate=True,
        solver="Euler",
        debug_logging=True,
        logger=print,  # fmi_call_logger=print,
        start_values={"pos[2]": 2},
    )
    plot_result(result)


def test_run_osp():
    sim = CosimExecution.from_step_size(step_size=1e7)  # empty execution object with fixed time step in nanos
    bb = CosimLocalSlave(fmu_path="./BouncingBall.fmu", instance_name="bb")

    print("SLAVE", bb, sim.status())

    ibb = sim.add_local_slave(bb)
    assert ibb == 0, f"local slave number {ibb}"

    reference_dict = {var_ref.name.decode(): var_ref.reference for var_ref in sim.slave_variables(ibb)}

    # Set initial values
    sim.real_initial_value(ibb, reference_dict["pos[2]"], 2.0)

    sim_status = sim.status()
    assert sim_status.current_time == 0
    assert CosimExecutionState(sim_status.state) == CosimExecutionState.STOPPED
    infos = sim.slave_infos()
    print("INFOS", infos)

    # Simulate for 1 second
    sim.simulate_until(target_time=3e9)


if __name__ == "__main__":
    retcode = pytest.main(["-rA", "-v", __file__])
    assert retcode == 0, f"Non-zero return code {retcode}"
    # test_bouncing_ball_class()
    # test_make_bouncing_ball()
    # test_use_fmu()
    # test_run_osp()
