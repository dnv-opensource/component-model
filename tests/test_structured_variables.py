import logging
from math import asin, pi
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from examples.axle import Axle
from fmpy.simulation import simulate_fmu
from fmpy.util import fmu_info
from fmpy.validation import validate_fmu

from component_model.model import Model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def check_pos(
    time: float, bi: list[float], pos0: list[np.ndarray], pos1: list[np.ndarray], a: float, eps: float = 1e-3
):
    """Check the new position with respect to the line integral and axle length."""
    a0 = np.linalg.norm(pos0[0] - pos0[1])
    assert abs(a0 - a) < eps, f"Axle length {a0} wrong at start. Expected: {a}"
    a1 = np.linalg.norm(pos1[0] - pos1[1])
    assert abs(a1 - a) < eps, f"Axle length {a1}wrong at end. Expected: {a}"
    s0_dot_s1 = np.dot(pos1[0] - pos0[0], pos1[1] - pos0[1])
    si = [np.linalg.norm(pos1[i] - pos0[i]) for i in range(2)]  # secant vectors for distance traveled
    if (si[0] < 0 and s0_dot_s1 > 0) or (si[0] > 0 and s0_dot_s1 < 0):
        si[0] = -si[0]
    if abs(si[0] - si[1]) < 1e-10:  # straight line (infinite radius)
        for i in range(2):
            assert abs(si[i] - abs(bi[i])) < eps, f"Length @{time}. {abs(bi[i])} != {si[i]}"
    else:
        ri = [a * si[i] / (si[1] - si[0]) for i in range(2)]
        if abs(si[0]) > abs(si[1]):  # ensure accuracy
            alpha = asin(si[0] / 2 / ri[0]) * 2
        else:
            alpha = asin(si[1] / 2 / ri[1]) * 2
        _bi = [alpha * ri[i] for i in range(2)]
        for i in range(2):
            assert abs(abs(_bi[i]) - abs(bi[i])) < eps, f"Arc length @{time}. {abs(bi[i])} != {abs(_bi[i])}"


def test_axle_class(show: bool):
    def do_drive(axle: Axle, t: float, dt: float):
        """Drive one time step and do checks."""
        pos0 = [np.copy(axle.wheels[i].pos) for i in range(2)]
        axle.drive(t, dt)
        check_pos(
            t,
            [pi * axle.wheels[i].diameter * axle.wheels[i].motor.rpm * dt for i in range(2)],
            pos0,
            [axle.wheels[i].pos for i in range(2)],
            axle.a,
        )

    axle = Axle()
    axle.init_drive()
    # go straight
    dt = 0.01
    for t in np.linspace(0.0, 1, int(1.0 / dt)):
        do_drive(axle, t, dt)
    _pos = (0.0, pi)
    assert abs(axle.wheels[0].track[1][-1] - _pos[1]) < 1e-10

    # turn left
    axle.wheels[1].motor.rpm = -1.1
    for t in np.linspace(1.0, 2.0, int(1.0 / dt)):
        do_drive(axle, t, dt)

    # turn right
    axle.wheels[1].motor.rpm = -0.9
    for t in np.linspace(2.0, 3.0, int(1.0 / dt)):
        do_drive(axle, t, dt)

    # turn motors against each other (pirouette)
    axle.wheels[1].motor.rpm = 1.0
    for t in np.linspace(3.0, 4.0, int(1.0 / dt)):
        do_drive(axle, t, dt)

    axle.wheels[1].motor.rpm = -1.0
    axle.wheels[0].motor.acc = 0.1  # accelerate with both motor0
    for t in np.linspace(4.0, 10.0, int(6.0 / dt)):
        axle.drive(t, dt)
    # print("TRACK", axle.wheels[0]._track, axle.wheels[1]._track)
    if show:
        axle.show()
        print("Positions", axle.wheels[0].pos, axle.wheels[1].pos)


@pytest.fixture(scope="session")
def axle_fmu():
    return _axle_fmu()


def _axle_fmu():
    """Make FMU and return .fmu file with path."""
    build_path = Path.cwd()
    build_path.mkdir(exist_ok=True)
    fmu_path = Model.build(
        script=str(Path(__file__).parent.parent / "examples" / "axle_fmu.py"),
        project_files=[Path(__file__).parent.parent / "examples" / "axle.py"],
        dest=build_path,
    )
    return fmu_path


def test_make_fmu(axle_fmu: Path, show: bool):
    info = fmu_info(filename=str(axle_fmu))  # this is a formatted string. Not easy to check
    if show:
        print(f"Info Oscillator: {info}")
    val = validate_fmu(filename=str(axle_fmu))
    assert not len(val), f"Validation of of {axle_fmu.name} was not successful. Errors: {val}"


def test_use_fmu(axle_fmu: Path, show: bool):
    """Test single FMUs."""
    # sourcery skip: move-assign
    result = simulate_fmu(
        axle_fmu,
        stop_time=10.0,
        step_size=0.01,
        validate=True,
        solver="Euler",
        debug_logging=True,
        logger=print,  # fmi_call_logger=print,
        start_values={
            "wheels[0].motor.rpm": -1.0,
            "wheels[1].motor.rpm": -1.0,
            "der(wheels[1].motor.rpm)": 0.5,
        },
        step_finished=None,  # pyright: ignore[reportArgumentType]  # (typing incorrect in fmpy)
        fmu_instance=None,  # pyright: ignore[reportArgumentType]  # (typing incorrect in fmpy)
    )
    if show:
        data: list[list[float]] = [[], [], [], []]
        for row in result:
            for i in range(4):
                data[i].append(row[i + 1])
        fig, ax = plt.subplots()
        ax.plot(data[0], data[1], label="1")
        ax.plot(data[2], data[3], label="2")
        ax.legend()
        plt.show()


if __name__ == "__main__":
    retcode = pytest.main(["-rP -s -v", __file__])
    assert retcode == 0, f"Return code {retcode}"
    import os

    os.chdir(Path(__file__).parent.absolute() / "test_working_directory")
    # test_axle_class(show=True)
    # test_make_fmu(_axle_fmu(), show=False)#True)
    # test_use_fmu(_axle_fmu(), show=True)
