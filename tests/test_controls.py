import logging

import matplotlib.pyplot as plt
import numpy as np
import pytest

from component_model.utils.controls import Controls

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def test_limits():
    _b = Controls(
        ("len", "polar", "azimuth"),  # bcrane boom control
        (((1, 20), None, (100, 100)), ((-2, 2), (-1, 1), (0, 0)), ((-1, 1), (-2, 2), (0, 0))),
    )
    assert _b.limits(0, 1) == _b.limits("len", 1) == (float("-inf"), float("inf")), (
        f"Found {_b.limits(0, 1)}, {_b.limits('len', 1)}"
    )
    assert _b.limits(1, 2) == _b.limits("polar", 2) == (0, 0), f"Found {_b.limits(1, 2)}, {_b.limits('polar', 2)}"
    with pytest.raises(AssertionError) as err:
        _b.append("additional", ((1, 0), (0, 1), 0, 1))
    assert err.value.args[0].startswith("Wrong order of limits:")
    # Controls can also be built step by step:
    _b = Controls()
    _b.append("len", ((1, 20), None, (1, 100)))
    _b.append("polar", ((-2, 2), 1))  # fixed velocity and the acceleration limits are not needed
    _b.append("azimuth", ((-1, 1), (-2, 2), (0, 0)))
    assert _b.limit(1, 2, 0) == _b.limit(1, 2, 1) == 0.0, "No polar acceleration allowed"
    assert _b.nogoals, "No goals yet set"
    # check values just beyond limits (caused by e.g. numerical issues), which should not lead to messages
    assert _b.check_limit(0, 0, 1 - 1e-14) == 1.0, "Minor correction"
    assert _b.check_limit(0, 0, 20 + 1e-14) == 20.0, "Minor correction"
    assert _b.check_limit(0, 0, 10 + 1e-10) == 10 + 1e-10, f"No correction: {_b.check_limit(0, 0, 10 + 1e-10)}"

    # try to set goal outside limits
    _b.limit_err = logging.CRITICAL
    with pytest.raises(ValueError) as err:  # type: ignore[assignment]  #it is a 'ValueError'
        _b.setgoal(1, 2, 9.9, 0.0)
    assert err.value.args[0] == "Goal 'polar'@ 9.9 is above the limit 0.0. Stopping execution."
    _b.limit_err = logging.WARNING
    _b.setgoal(1, 2, 9.9, 0.0)
    assert _b.nogoals, f"No goals expected, because the adjusted goal is already reached. Found {_b.goals}"
    _b.setgoal(0, 2, 1.1, 0.0)
    assert _b.goals == [((float("inf"), 1.1),), None, None], f"Found {_b.goals}"
    assert not _b.nogoals, "At least one goal set"


def test_goal(show: bool = False):
    def do_goal(order: int, value: float, current: np.ndarray | None = None, t_end: float = 10.0):
        time = 0.0
        if current is not None:
            _b.current[0] = current
        if order == 2:
            _b.setgoal("len", order, value, time)
        elif order == 1:
            _b.setgoal("len", order, value, time)
        else:
            _b.setgoal("len", order, value, time)
        dt = 0.1
        res: list[tuple] = []
        assert _b.goals[0] is not None
        res.append((time, _b.current[0][0], _b.current[0][1], _b.goals[0][0][1]))
        while time + dt < t_end:
            _b.step(time, dt)
            time += dt
            res.append((time, *_b.current[0]))
        if show:
            do_show(res)
        return res

    _b = Controls(limit_err=logging.CRITICAL)
    _b.append("len", ((-100.0, 90.0), None, (-1.0, 0.5)))
    _b.append("polar", ((-2.0, 2.0), 1.0))  # fixed velocity and the acceleration limits are not needed
    _b.append("azimuth", ((-1.0, 1.0), (-2.0, 2.0), (0.0, 0.0)))

    # get from one position to another (zero velocity and acceleration on both ends)
    res = do_goal(0, 0.0, current=np.array((10.0, 0.0, 0.0), float), t_end=11.7)
    assert abs(_b.current[0][0] + 12.5) < 1e-13, f"Found {_b.current[0][0]}"
    assert abs(_b.current[0][1]) < 1e-13
    assert abs(_b.current[0][2]) < 1e-13

    # get from one position to another (non-zero velocity on start)
    res = do_goal(0, 0.0, current=np.array((10.0, 1.0, 0.0), float), t_end=3.65)
    assert np.allclose(_b.current[0], (9.375, 0.0, 0.0)), f"Found {_b.current[0]}"

    # get from one velocity to another
    res = do_goal(1, -2.0, current=np.array((10.0, 1.0, 0.0), float), t_end=3.1)
    assert np.allclose(_b.current[0], (8.5, -2.0, 0.0))

    # set an acceleration (non-zero position and velocity)
    res = do_goal(2, -0.1, current=np.array((10.0, 1.0, 0.0), float), t_end=2.01)
    expected = (10 + 1.0 * 2.0 - 0.5 * 0.1 * 2.0**2, 1.0 - 0.1 * 2.0, -0.1)
    assert np.allclose(_b.current[0], expected), f"{_b.current[0]} != {expected} "

    _b.limit_err = logging.WARNING  # allow corrections from now on
    _b.current = [np.array((0.0, 0.0, 0.0), float)] * 3

    # accelerate in 10 time units
    res = do_goal(2, 1.1)
    for t, x, v, a in res:
        assert np.allclose((x, v, a), (0.5 * 0.5 * t * t, 0.5 * t, 0.5)), f"@{t}: Found {(x, v, a)}"

    assert abs(_b.current[0][1] - 5.0) < 1e-9, f"Found {_b.current[0][1]}"
    assert abs(_b.current[0][0] - 25.0) < 1e-9, f"Found {_b.current[0][0]}"

    # Speed from 5 to 2.2
    res = do_goal(1, 2.2)
    tgoal = (2.2 - 5) / (-1.0)
    for t, x, v, a in res:
        if t < tgoal:
            assert abs(a + 1) < 1e-9, f"@{t} Acc.: {a}"
            assert abs(v - 5.0 - a * t) < 1e-9, f"@{t} Velocity: {v}"
            assert abs(x - 25.0 - 5 * t - 0.5 * a * t * t) < 1e-9, f"@{t} Pos.: {x} != {25 + 5 * t - 0.5 * 1 * t * t}"
        else:
            _x = 25.0 + 5.0 * tgoal - 0.5 * 1.0 * tgoal**2 + 2.2 * (t - tgoal)
            assert np.allclose((x, v, a), (_x, 2.2, 0.0)), f"@{t}: {(x, v, a)} != {(_x, 2.2, 0)}"

    res = do_goal(0, 0.0)


def do_show(results: list[tuple]):
    """Plot selected traces."""
    times = [row[0] for row in results]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.set_size_inches(10, 6)
    _ = ax1.plot(times, [row[1] for row in results], label="position")
    _ = ax2.plot(times, [row[2] for row in results], label="velocity")
    _ = ax3.plot(times, [row[3] for row in results], label="acceleration")
    _ = ax1.legend()
    _ = ax2.legend()
    _ = ax3.legend()
    plt.show()


if __name__ == "__main__":
    retcode = pytest.main(["-rA", "-v", "--rootdir", "../", "--show", "False", __file__])
    assert retcode == 0, f"Non-zero return code {retcode}"
    logging.basicConfig(level=logging.DEBUG)
    plt.set_loglevel(level="warning")
    # test_limits()
    # test_goal(show=True)
