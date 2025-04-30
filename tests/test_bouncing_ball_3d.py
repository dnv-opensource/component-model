# pyright: reportAttributeAccessIssue=false
# pyright: reportOptionalMemberAccess=false
import time
import xml.etree.ElementTree as ET  # noqa: N817
from math import sqrt
from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
import pytest
from fmpy import plot_result, simulate_fmu  # type: ignore
from fmpy.util import fmu_info  # type: ignore
from fmpy.validation import validate_fmu  # type: ignore
from libcosimpy.CosimEnums import CosimErrorCode, CosimExecutionState
from libcosimpy.CosimExecution import CosimExecution
from libcosimpy.CosimManipulator import CosimManipulator
from libcosimpy.CosimObserver import CosimObserver
from libcosimpy.CosimSlave import CosimLocalSlave
from pythonfmu.default_experiment import DefaultExperiment

from component_model.model import Model
from component_model.utils.fmu import model_from_fmu


def _in_interval(x: float, x0: float, x1: float):
    return x0 <= x <= x1 or x1 <= x <= x0


def arrays_equal(arr1, arr2, eps=1e-7):
    assert len(arr1) == len(arr2), "Length not equal!"

    for i in range(len(arr1)):
        # assert type(arr1[i]) == type(arr2[i]), f"Array element {i} type {type(arr1[i])} != {type(arr2[i])}"
        assert abs(arr1[i] - arr2[i]) < eps, f"Component {i}: {arr1[i]} != {arr2[i]}"


def _to_et(file: str, sub: str = "modelDescription.xml"):
    with ZipFile(file) as zp:
        xml = zp.read(sub)
    return ET.fromstring(xml)


def do_show(result: list):
    fig, ax = plt.subplots()
    ax.plot([res[3] for res in result], label="z-position")
    ax.plot([res[4] for res in result], label="x-speed")
    ax.plot([res[6] for res in result], label="z-speed")
    ax.legend()
    plt.show()


@pytest.fixture(scope="session")
def bouncing_ball_fmu():
    return _bouncing_ball_fmu()


def _bouncing_ball_fmu():
    build_path = Path.cwd()
    build_path.mkdir(exist_ok=True)
    fmu_path = Model.build(
        str(Path(__file__).parent.parent / "examples" / "bouncing_ball_3d.py"),
        project_files=[],
        dest=build_path,
    )
    return fmu_path


def test_bouncing_ball_class(show):
    """Test the BouncingBall3D class in isolation.

    The first four lines are necessary to ensure that the BouncingBall3D class can be accessed:
    If pytest is run from the command line, the current directory is the package root,
    but when it is run from the editor (__main__) it is run from /tests/.
    """
    from examples.bouncing_ball_3d import BouncingBall3D

    bb = BouncingBall3D()
    assert bb._pos.display is not None
    assert bb._pos.setter is not None
    assert bb._pos.getter is not None
    assert bb._speed.getter is not None
    assert bb._p_bounce.getter is not None

    result = []

    def get_result():
        """Make a row of the fmpy results vector (all output variables in display units)"""
        _pos = bb._pos.getter()
        assert isinstance(_pos, list)
        _speed = bb._speed.getter()
        assert isinstance(_speed, list)
        _p_bounce = bb._p_bounce.getter()
        assert isinstance(_p_bounce, list)
        result.append((bb.time, *_pos, *_speed, *_p_bounce))

    h_fac = 1.0
    if len(bb._pos.display) > 1 and bb._pos.display[2] is not None:  # the main test settings
        arrays_equal(bb.pos, (0, 0, 10 * 0.0254))  # was provided as inch
        arrays_equal(bb.speed, (1, 0, 0))
        assert bb.g == 9.81
        assert bb.e == 0.9
        h_fac = 0.0254
    h0 = bb.pos[2]
    t_bounce = sqrt(2 * h0 / bb.g)
    v_bounce = bb.g * t_bounce  # speed in z-direction
    x_bounce = bb.speed[0] * t_bounce  # x-position where it bounces in m
    time = 0
    assert isinstance(bb.default_experiment, DefaultExperiment)
    dt = bb.default_experiment.step_size
    assert dt == 0.01
    # set start values (in display units. Are translated to internal units
    if len(bb._pos.display) > 1 and bb._pos.display[2] is not None:
        bb._pos.setter((0, 0, 10))
    t_b, p_b = bb.next_bounce()
    assert t_bounce == t_b
    # print("Bounce", t_bounce, x_bounce, p_b)
    arrays_equal((x_bounce, 0, 0), p_b), f"x_bounce:{x_bounce} != {p_b[0]}"  # type: ignore ##??
    get_result()
    # after one step
    bb.do_step(time, dt)
    get_result()
    # print("After one step", result(bb))
    arrays_equal(
        result[-1],
        (
            0.01,  # time
            0.01,  # pos
            0,
            (h0 - 0.5 * bb.g * 0.01**2) / h_fac,
            1,  # speed
            0,
            -bb.g * 0.01,
            x_bounce,  # p_bounce
            0,
            0,
        ),
    )
    # just before bounce
    t_before = int(t_bounce / dt) * dt  # just before bounce
    if t_before == t_bounce:  # at the interval border
        t_before -= dt
    for _ in range(int(t_before / dt) - 1):
        bb.do_step(time, dt)
        get_result()
    # print(f"Just before bounce @{t_bounce}, {t_before}: {result[-1]}")
    arrays_equal(
        result[-1],
        (
            t_before,
            1 * t_before,
            0,
            (h0 - 0.5 * bb.g * t_before**2) / h_fac,
            1,
            0,
            -bb.g * t_before,
            x_bounce,
            0,
            0,
        ),
        eps=0.003,
    )
    # just after bounce
    # print(f"Step {len(z)}, time {bb.time}, pos:{bb.pos}, speed:{bb.speed}, t_bounce:{bb.t_bounce}, p_bounce:{bb.p_bounce}")
    bb.do_step(time, dt)
    get_result()
    ddt = t_before + dt - t_bounce  # time from bounce to end of step
    x_bounce2 = x_bounce + 2 * v_bounce * bb.e * 1.0 * bb.e / bb.g
    arrays_equal(
        result[-1],
        (
            t_before + dt,
            t_bounce * 1 + 1 * bb.e * ddt,
            0,
            (v_bounce * bb.e * ddt - 0.5 * bb.g * ddt**2) / h_fac,
            bb.e * 1,
            0,
            (v_bounce * bb.e - bb.g * ddt),
            x_bounce2,
            0,
            0,
        ),
        eps=0.03,
    )
    # from bounce to bounce
    v_x, v_z, t_b, x_b = (
        1.0,
        v_bounce,
        t_bounce,
        x_bounce,
    )  # set start values (first bounce)
    # print(f"1.bounce time: {t_bounce} v_x:{v_x}, v_z:{v_z}, t_b:{t_b}, x_b:{x_b}")
    for _n in range(2, 100):  # from bounce to bounce
        v_x = v_x * bb.e  # adjusted speeds
        v_z = v_z * bb.e
        delta_t = 2 * v_z / bb.g  # time for one bounce (parabola): v(t) = v0 - g*t/2 => 2*v0/g = t
        t_b += delta_t
        x_b += v_x * delta_t
        # print(f"Bounce {n} @{t_b}")
        while bb.time <= t_b:
            # print(f"Step {len(z)}, time {bb.time}, pos:{bb.pos}, speed:{bb.speed}, t_bounce:{bb.t_bounce}, p_bounce:{bb.p_bounce}")
            bb.do_step(time, dt)
            get_result()
        # print( f"Bounce {n}: {bb.pos}, steps:{len(result)}, v_x:{v_x}, v_z:{v_z}, delta_t:{delta_t}, t_b:{t_b}, x_b:{x_b}")
        assert abs(bb.pos[2]) < 1e-2, f"z-position {bb.pos[2]} should be close to 0"
        if delta_t > 2 * dt:
            assert isinstance(result[-2][6], float) and isinstance(result[-1][6], float)
            assert result[-2][6] < 0.0 and result[-1][6] > 0.0, (
                f"Expected speed sign change {result[-2][6]}-{result[-1][6]}when bouncing"
            )
            assert bb.speed[0] == result[-2][4] * bb.e, "Reduced speed in x-direction"
    if show:
        do_show(result)


def test_make_bouncing_ball(bouncing_ball_fmu):
    _ = fmu_info(bouncing_ball_fmu)  # not necessary, but it lists essential properties of the FMU
    et = _to_et(bouncing_ball_fmu)
    assert et.attrib["fmiVersion"] == "2.0", "FMI Version"
    # similarly other critical issues of the modelDescription can be checked
    assert et.attrib["variableNamingConvention"] == "structured", "Variable naming convention. => use [i] for arrays"
    #    print(et.attrib)
    val = validate_fmu(str(bouncing_ball_fmu))
    assert not len(val), (
        f"Validation of the modelDescription of {bouncing_ball_fmu.name} was not successful. Errors: {val}"
    )


def test_use_fmu(bouncing_ball_fmu, show):
    """Test and validate the basic BouncingBall using fmpy and not using OSP or case_study."""
    assert bouncing_ball_fmu.exists(), f"File {bouncing_ball_fmu} does not exist"
    dt = 0.01
    result = simulate_fmu(  # type: ignore[reportArgumentType]
        bouncing_ball_fmu,
        start_time=0.0,
        stop_time=3.0,
        step_size=dt,
        validate=True,
        solver="Euler",
        debug_logging=False,
        visible=True,
        logger=print,  # fmi_call_logger=print,
        start_values={
            "pos[2]": 10.0,
            "speed[0]": 1.0,
            "e": 0.9,
            "g": 9.81,
        },
    )
    if show:
        plot_result(result)
    h0 = 10 * 0.0254
    g = 9.81
    e = 0.9
    h_fac = 0.0254
    t_bounce = sqrt(2 * h0 / g)
    v_bounce = g * t_bounce  # speed in z-direction
    x_bounce = t_bounce / 1.0  # x-position where it bounces in m
    # Note: default values are reported at time 0!
    arrays_equal(list(result[0])[:7], [0, 0, 0, 10, 1, 0, 0])  # time,pos-3, speed-3(, p_bounce-3 not calculated)
    # print(f"Result[1]: {result[1]}")
    arrays_equal(
        result[1],
        (
            0.01,  # time
            0.01,  # pos
            0,
            (h0 - 0.5 * g * 0.01**2) / h_fac,
            1,  # speed
            0,
            -g * 0.01,
            x_bounce,  # p_bounce
            0,
            0,
        ),
    )
    arrays_equal(
        result[1],
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
    if t_before == t_bounce:  # at the interval border
        t_before -= dt
    # print(f"Just before bounce @{t_bounce}, {t_before}: {result[-1]}")
    arrays_equal(
        result[int(t_before / dt)],
        (
            t_before,
            1 * t_before,
            0,
            (h0 - 0.5 * g * t_before**2) / h_fac,
            1,
            0,
            -g * t_before,
            x_bounce,
            0,
            0,
        ),
        eps=0.003,
    )
    # just after bounce
    # print(f"Step {len(z)}, time {bb.time}, pos:{bb.pos}, speed:{bb.speed}, t_bounce:{bb.t_bounce}, p_bounce:{bb.p_bounce}")
    ddt = t_before + dt - t_bounce  # time from bounce to end of step
    x_bounce2 = x_bounce + 2 * v_bounce * e * 1.0 * e / g
    arrays_equal(
        result[int((t_before + dt) / dt)],
        (
            t_before + dt,
            t_bounce * 1 + 1 * e * ddt,
            0,
            (v_bounce * e * ddt - 0.5 * g * ddt**2) / h_fac,
            e * 1,
            0,
            (v_bounce * e - g * ddt),
            x_bounce2,
            0,
            0,
        ),
        eps=0.03,
    )
    # from bounce to bounce
    v_x, v_z, t_b, x_b = (
        1.0,
        v_bounce,
        t_bounce,
        x_bounce,
    )  # set start values (first bounce)
    row = int((t_before + dt) / dt)
    # print(f"1.bounce time: {t_bounce} v_x:{v_x}, v_z:{v_z}, t_b:{t_b}, x_b:{x_b}")
    for n in range(2, 100):  # from bounce to bounce
        v_x = v_x * e  # adjusted speeds
        v_z = v_z * e
        delta_t = 2 * v_z / g  # time for one bounce (parabola): v(t) = v0 - g*t/2 => 2*v0/g = t
        t_b += delta_t
        x_b += v_x * delta_t
        print(f"Bounce {n} @{t_b}")
        while result[row][0] <= t_b:  # spool to the time just after the bounce
            row += 1
            if row >= len(result):
                return
        # print( f"Bounce {n}: {result[row][3]}, steps:{row}, v_x:{v_x}, v_z:{v_z}, delta_t:{delta_t}, t_b:{t_b}, x_b:{x_b}")
        assert abs(min(result[row - 1][3], result[row][3])) < 0.3, f"z-position {result[row][3]} should be close to 0"
        if delta_t > 2 * dt:
            assert result[row - 1][6] < 0 and result[row][6] > 0, (
                f"Expected speed sign change {result[row - 1][6]}-{result[row][6]}when bouncing"
            )
            assert abs(result[row - 1][4] * e - result[row][4]) < 1e-15, "Reduced speed in x-direction"


def test_from_osp(bouncing_ball_fmu):
    def get_status(sim):
        status = sim.status()
        return {
            "currentTime": status.current_time,
            "state": CosimExecutionState(status.state).name,
            "error_code": CosimErrorCode(status.error_code).name,
            "real_time_factor": status.real_time_factor,
            "rolling_average_real_time_factor": status.rolling_average_real_time_factor,
            "real_time_factor_target": status.real_time_factor_target,
            "is_real_time_simulation": status.is_real_time_simulation,
            "steps_to_monitor": status.steps_to_monitor,
        }

    sim = CosimExecution.from_step_size(step_size=1e7)  # empty execution object with fixed time step in nanos
    bb = CosimLocalSlave(fmu_path=str(bouncing_ball_fmu.absolute()), instance_name="bb")

    ibb = sim.add_local_slave(bb)
    assert ibb == 0, f"local slave number {ibb}"
    info = sim.slave_infos()
    assert info[0].name.decode() == "bb", "The name of the component instance"
    assert info[0].index == 0, "The index of the component instance"
    assert sim.slave_index_from_instance_name("bb") == 0
    assert sim.num_slaves() == 1
    assert sim.num_slave_variables(0) == 11, "3*pos, 3*speed, g, e, 3*p_bounce"
    variables = {var_ref.name.decode(): var_ref.reference for var_ref in sim.slave_variables(ibb)}
    assert variables == {
        "pos[0]": 0,
        "pos[1]": 1,
        "pos[2]": 2,
        "speed[0]": 3,
        "speed[1]": 4,
        "speed[2]": 5,
        "g": 6,
        "e": 7,
        "p_bounce[0]": 8,
        "p_bounce[1]": 9,
        "p_bounce[2]": 10,
    }

    # Set initial values
    sim.real_initial_value(ibb, variables["g"], 1.5)  # actual setting will only happen after start_initialization_mode

    assert get_status(sim)["state"] == "STOPPED"

    observer = CosimObserver.create_last_value()
    assert sim.add_observer(observer)
    manipulator = CosimManipulator.create_override()
    assert sim.add_manipulator(manipulator)

    values = observer.last_real_values(0, list(range(11)))
    assert values == [0.0] * 11, "No initial values yet! - as expected"

    # that does not seem to work (not clear why):    assert sim.step()==True
    assert sim.simulate_until(target_time=1e7), "Simulate for one base step did not work"
    assert get_status(sim)["currentTime"] == 1e7, "Time after simulation not correct"
    values = observer.last_real_values(0, list(range(11)))
    assert values[6] == 1.5, "Initial setting did not work"
    assert values[5] == -0.015, "Initial setting did not have the expected effect on speed"


#     values = observer.last_real_values(0, list(range(11)))
#     print("VALUES2", values)
#
#     manipulator.reset_variables(0, CosimVariableType.REAL, [6])

#    sim.simulate_until(target_time=3e9)


def test_from_fmu(bouncing_ball_fmu):
    assert bouncing_ball_fmu.exists(), "FMU not found"
    model = model_from_fmu(bouncing_ball_fmu)
    assert model["name"] == "BouncingBall3D", f"Name: {model['name']}"
    assert (
        model["description"] == "Another Python-based BouncingBall model, using Model and Variable to construct a FMU"
    )
    assert model["author"] == "DNV, SEACo project"
    assert model["version"] == "0.1"
    assert model["license"].startswith("Permission is hereby granted, free of charge, to any person obtaining a copy")
    assert model["copyright"] == f"Copyright (c) {time.localtime()[0]} DNV, SEACo project", (
        f"Found: {model['copyright']}"
    )
    assert model["default_experiment"] is not None
    assert (
        model["default_experiment"]["start_time"],
        model["default_experiment"]["step_size"],
        model["default_experiment"]["stop_time"],
    ) == (0.0, 0.01, 1.0)


if __name__ == "__main__":
    retcode = 0  # pytest.main(["-rA", "-v", "--rootdir", "../", "--show", "False", __file__])
    assert retcode == 0, f"Non-zero return code {retcode}"
    import os

    os.chdir(Path(__file__).parent / "test_working_directory")
    # test_bouncing_ball_class(show=False)
    test_make_bouncing_ball(_bouncing_ball_fmu())
    # test_use_fmu(_bouncing_ball_fmu(), True)
    # test_from_fmu( _bouncing_ball_fmu())
    # test_from_osp( _bouncing_ball_fmu())
