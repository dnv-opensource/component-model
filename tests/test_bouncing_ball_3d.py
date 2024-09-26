import time
import xml.etree.ElementTree as ET  # noqa: N817
from math import sqrt
from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
import pytest
from component_model.model import Model  # type: ignore
from component_model.utils import model_from_fmu
from fmpy import simulate_fmu  # type: ignore
from fmpy.util import fmu_info  # type: ignore
from fmpy.validation import validate_fmu  # type: ignore
from libcosimpy.CosimEnums import CosimExecutionState
from libcosimpy.CosimExecution import CosimExecution
from libcosimpy.CosimSlave import CosimLocalSlave

from tests.examples.bouncing_ball_3d import BouncingBall3D  # type: ignore


def _in_interval(x: float, x0: float, x1: float):
    return x0 <= x <= x1 or x1 <= x <= x0


def arrays_equal(arr1, arr2, eps=1e-7):
    assert len(arr1) == len(arr2), "Length not equal!"

    for i in range(len(arr1)):
        # assert type(arr1[i]) == type(arr2[i]), f"Array element {i} type {type(arr1[i])} != {type(arr2[i])}"
        assert abs(arr1[i] - arr2[i]) < eps, f"Component {i}: {arr1[i]} != {arr2[i]}"


def show_data(_z):
    fig, ax = plt.subplots()
    ax.plot(_z)
    plt.title("Data (_z)", loc="left")
    plt.show()


def _to_et(file: str, sub: str = "modelDescription.xml"):
    with ZipFile(file) as zp:
        xml = zp.read(sub)
    return ET.fromstring(xml)


def result(bb):
    """Make a row of the fmpy results vector (all output variables in display units)"""
    return (bb.time, *bb._pos.getter(), *bb._speed.getter(), *bb._p_bounce.getter())


@pytest.fixture(scope="session")
def bouncing_ball_fmu():
    build_path = Path.cwd() / "fmus"
    build_path.mkdir(exist_ok=True)
    fmu_path = Model.build(
        str(Path(__file__).parent / "examples" / "bouncing_ball_3d.py"),
        project_files=[],
        dest=build_path,
    )
    return fmu_path


def test_bouncing_ball_class():
    bb = BouncingBall3D(
        pos=("0 m", "0 m", "10 inch"), speed=("1 m/s", "0 m/s", "0 m/s"), g="9.81 m/s^2", e=0.9, min_speed_z=1e-6
    )
    arrays_equal(bb.pos, (0, 0, 10 * 0.0254))  # was provided as inch
    arrays_equal(bb.speed, (1, 0, 0))
    assert bb.g == 9.81
    assert bb.e == 0.9
    t_bounce = sqrt(2 * 10 * 0.0254 / 9.81)
    v_bounce = 9.81 * t_bounce  # speed in z-direction
    x_bounce = t_bounce  # x-position where it bounces in m
    arrays_equal(bb.p_bounce, (x_bounce, 0, 0))
    time = 0
    dt = bb.default_experiment["stepSize"]
    assert dt == 0.01
    # set start values (in display units. Are translated to internal units
    bb._pos.setter((0, 0, 10))
    t_b, p_b = bb.next_bounce()
    assert t_bounce == t_b
    arrays_equal((x_bounce, 0, 0), p_b), f"x_bounce:{x_bounce} != {p_b[0]}"
    z = [bb._pos.getter()[2]]
    # after one step
    bb.do_step(time, dt)
    z.append(bb._pos.getter()[2])
    arrays_equal(
        result(bb),
        (
            0.01,
            0.01,
            0,
            (10 * 0.0254 - 0.5 * 9.81 * 0.01**2) / 0.0254,
            1,
            0,
            -9.81 * 0.01,
            sqrt(2 * 10 * 0.0254 / 9.81),
            0,
            0,
        ),
    )
    # just before bounce
    t_before = int(t_bounce / dt) * dt  # just before bounce
    assert t_before == 0.22
    for _ in range(int(t_bounce / dt) - 1):
        bb.do_step(time, dt)
        z.append(bb._pos.getter()[2])
    arrays_equal(
        result(bb),
        (
            t_before,
            1 * t_before,
            0,
            (10 * 0.0254 - 0.5 * 9.81 * t_before**2) / 0.0254,
            1,
            0,
            -9.81 * t_before,
            x_bounce,
            0,
            0,
        ),
        eps=0.003,
    )
    # just after bounce
    bb.do_step(time, dt)
    z.append(bb._pos.getter()[2])
    ddt = t_before + dt - t_bounce  # time from bounce to end of step
    x_bounce2 = x_bounce + 2 * v_bounce * 0.9 * 1.0 * 0.9 / 9.81
    arrays_equal(
        result(bb),
        (
            t_before + dt,
            t_bounce * 1 + 1 * 0.9 * ddt,
            0,
            (v_bounce * 0.9 * ddt - 0.5 * 9.81 * ddt**2) / 0.0254,
            0.9 * 1,
            0,
            (v_bounce * 0.9 - 9.81 * ddt),
            x_bounce2,
            0,
            0,
        ),
        eps=0.03,
    )


#     # from bounce to bounce
#     v_x, v_z, t_b, x_b = 1.0, v_bounce, t_bounce, x_bounce
#     for n in range(2, 100): # from bounce to bounce
#         v_x = v_x* 0.9
#         v_z = v_z* 0.9
#         delta_t = 2* v_z* v_x/ 9.81
#         t_b = t_b + delta_t
#         x_b = x_b + v_x* t_b
#
#         for _ in range( int( delta_t/dt)):
#             bb.do_step( time, dt)
#             z.append( bb._pos.getter()[2])
#
#         print( f"Bounce {n}: {bb.pos}, steps:{len(z)}")
#    show_data(z)
#     return
#     arrays_equal(result[int(2.5 / dt)], (2.5, 0, 0), eps=0.4)
#     arrays_equal(result[int(3 / dt)], (3, 0, 0))
#     print("RESULT", result[int(t_before / dt) + 1])
#
#     time += dt
#     height.append(bb.pos[2])
#     times.append(time)


def test_make_bouncing_ball(bouncing_ball_fmu):
    _ = fmu_info(bouncing_ball_fmu)  # not necessary, but it lists essential properties of the FMU
    et = _to_et(bouncing_ball_fmu)
    assert et.attrib["fmiVersion"] == "2.0", "FMI Version"
    # similarly other critical issues of the modelDescription can be checked
    assert et.attrib["variableNamingConvention"] == "structured", "Variable naming convention. => use [i] for arrays"
    #    print(et.attrib)
    val = validate_fmu(str(bouncing_ball_fmu))
    assert not len(
        val
    ), f"Validation of the modelDescription of {bouncing_ball_fmu.name} was not successful. Errors: {val}"


def test_use_fmu(bouncing_ball_fmu):
    _ = simulate_fmu(
        str(bouncing_ball_fmu),
        stop_time=3.0,
        step_size=0.1,
        validate=True,
        solver="Euler",
        debug_logging=True,
        logger=print,  # fmi_call_logger=print,
        start_values={"pos[2]": 2},
    )


def test_run_osp(bouncing_ball_fmu):
    sim = CosimExecution.from_step_size(step_size=1e7)  # empty execution object with fixed time step in nanos
    bb = CosimLocalSlave(fmu_path=str(bouncing_ball_fmu.absolute()), instance_name="bb")

    ibb = sim.add_local_slave(bb)
    assert ibb == 0, f"local slave number {ibb}"

    reference_dict = {var_ref.name.decode(): var_ref.reference for var_ref in sim.slave_variables(ibb)}

    # Set initial values
    sim.real_initial_value(ibb, reference_dict["pos[2]"], 2.0)

    sim_status = sim.status()
    assert sim_status.current_time == 0
    assert CosimExecutionState(sim_status.state) == CosimExecutionState.STOPPED
    _ = sim.slave_infos()

    # Simulate for 1 second
    sim.simulate_until(target_time=3e9)


def test_from_fmu(bouncing_ball_fmu):
    assert bouncing_ball_fmu.exists(), "FMU not found"
    model = model_from_fmu(bouncing_ball_fmu)
    assert model["name"] == "BouncingBall3D", f"Name: {model['name']}"
    assert (
        model["description"]
        == "Another BouncingBall model, made in Python and using Model and Variable to construct a FMU"
    )
    assert model["author"] == "DNV, SEACo project"
    assert model["version"] == "0.1"
    assert model["license"].startswith("Permission is hereby granted, free of charge, to any person obtaining a copy")
    assert model["copyright"] == f"Copyright (c) {time.localtime()[0]} DNV, SEACo project", f"Found: {model.copyright}"
    assert model["default_experiment"] is not None
    assert (
        model["default_experiment"]["start_time"],
        model["default_experiment"]["step_size"],
        model["default_experiment"]["stop_time"],
    ) == (0.0, 0.01, 1.0)


if __name__ == "__main__":
    retcode = pytest.main(["-rA", "-v", __file__])
    assert retcode == 0, f"Non-zero return code {retcode}"
