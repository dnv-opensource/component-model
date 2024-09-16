import time
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

from component_model.utils import model_from_fmu


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

@pytest.fixture(scope="session")
def bouncing_ball_fmu(tmp_path_factory):
    build_path = tmp_path_factory.mktemp("fmu")
    fmu_path = Model.build("../component_model/example_models/bouncing_ball.py", project_files=[], dest=build_path)
    return fmu_path

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


def test_make_bouncing_ball(bouncing_ball_fmu):
    info = fmu_info(bouncing_ball_fmu)  # not necessary, but it lists essential properties of the FMU
    print(f"Info: {info}")
    et = _to_et(bouncing_ball_fmu)
    assert et.attrib["fmiVersion"] == "2.0", "FMI Version"
    # similarly other critical issues of the modelDescription can be checked
    assert et.attrib["variableNamingConvention"] == "structured", "Variable naming convention. => use [i] for arrays"
    #    print(et.attrib)
    val = validate_fmu(bouncing_ball_fmu)
    assert not len(val), f"Validation of the modelDescription of {bouncing_ball_fmu.name} was not successful. Errors: {val}"


def test_use_fmu(bouncing_ball_fmu):
    result = simulate_fmu(
        bouncing_ball_fmu,
        stop_time=3.0,
        step_size=0.1,
        validate=True,
        solver="Euler",
        debug_logging=True,
        logger=print,  # fmi_call_logger=print,
        start_values={"pos[2]": 2},
    )
    plot_result(result)


def test_run_osp(bouncing_ball_fmu):
    sim = CosimExecution.from_step_size(step_size=1e7)  # empty execution object with fixed time step in nanos
    bb = CosimLocalSlave(fmu_path=str(bouncing_ball_fmu.absolute()), instance_name="bb")

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

def test_from_fmu(bouncing_ball_fmu):
    assert bouncing_ball_fmu.exists(), "FMU not found"
    model = model_from_fmu(bouncing_ball_fmu)
    assert model.name == "BouncingBallFMU", f"Name:{model.name}"
    print( dir(model))
    assert model.description == "Simple bouncing ball test FMU", f"Description:{model.description}"
    assert model.author == "DNV, SEACo project"
    assert model.version == "0.1"
    assert model.license.startswith("Permission is hereby granted, free of charge, to any person obtaining a copy")
    assert model.copyright == f"Copyright (c) {time.localtime()[0]} DNV, SEACo project", f"Found: {model.copyright}"
    assert model.default_experiment is None
    assert (
        model.default_experiment.start_time,
        model.default_experiment.step_size,
        model.default_experiment.stop_time,
        model.default_experiment.tolerance,
    ) == (0.0, 0.1, 10.0, 0.001)
    assert model.flags == {
        "needsExecutionTool": True,
        "canHandleVariableCommunicationStepSize": True,
        "canNotUseMemoryManagementFunctions": True,
    }
    for idx, var in model.vars.items():
        print(idx, var)
    assert model.vars[0].name == "x[0]"
    assert model.vars[0].value0 == 0.0
    assert model.vars[6].name == "bounceFactor"
