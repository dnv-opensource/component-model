import logging
from functools import partial
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pytest

from component_model.utils.controls import RW, Control, Controls

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def do_show(results: list[tuple[float, ...]]):
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


def test_limits():
    _val = 1.0

    class _RW(RW):
        def __call__(self, v: float | None = None) -> float:
            if v is not None:
                _val = v
            return _val

    _b = Controls()
    _len = Control("len", ((1, 20), None, (100, 100)), _RW())
    _polar = Control("polar", ((-2, 2), (-1, 1), (0, 0)), _RW())
    _azimuth = Control("azimuth", ((-1, 1), (-2, 2), (0, 0)), _RW())
    _b.extend((_len, _polar, _azimuth))
    assert _b[0].name == "len", f"Control 'len' expected. Found {_b[0].name}"
    assert _b[0].limits(1) == _b["len"].limits(1) == (float("-inf"), float("inf")), (
        f"Found {_b[0].limits(1)}, {_b['len'].limits(1)}"
    )
    assert _b[1].limits(2) == _b["polar"].limits(2) == (0, 0), f"Found {_b[1].limits(2)}, {_b['polar'].limits(2)}"
    with pytest.raises(AssertionError) as err:
        _b.append(Control("additional", ((1, 0), (0, 1), (0, 1)), _RW()))
    assert err.value.args[0].startswith("Wrong order of limits:")
    _b["len"].limits(2, (1, 100))
    # fixed velocity and the acceleration limits are not needed:
    _b.append(Control("newPolar", ((-2, 2), 1), _RW()))
    assert _b["newPolar"].limit(2, 0) == _b[3].limit(2, 1) == 0.0, "No polar acceleration allowed"
    assert _b.nogoals, "No goals yet set"
    # check values just beyond limits (caused by e.g. numerical issues), which should not lead to messages
    assert _b[0].check_limit(0, 1 - 1e-14) == 1.0, "Minor correction"
    assert _b[0].check_limit(0, 20 + 1e-14) == 20.0, "Minor correction"
    assert _b[0].check_limit(0, 10 + 1e-10) == 10 + 1e-10, f"No correction: {_b[0].check_limit(0, 10 + 1e-10)}"

    # try to set goal outside limits
    Controls.limit_err = logging.CRITICAL
    with pytest.raises(ValueError) as err:  # type: ignore[assignment] ## mypy believes that it is an AssertionError??
        _b[1].setgoal(2, 9.9)
    assert err.value.args[0] == "Goal 'polar'@ 9.9 is above the limit 0. Stopping execution."
    Controls.limit_err = logging.WARNING
    _b[1].setgoal(2, 9.9, 0.0)
    assert _b.nogoals, "No goals expected, because the adjusted goal is already reached."
    _b[0].setgoal(2, 1.1, 0.0)
    assert _b[0].goal == [(float("inf"), 1.1)], f"Found {_b[0].goal}"
    assert not _b.nogoals, "At least one goal set"


def test_goal(show: bool = False):
    def do_goal(
        order: int,
        value: float,
        current: np.ndarray | None = None,
        t_end: float = 10.0,
        change: tuple[float, int, float] | None = None,
    ):
        if change is not None:
            t1, order1, val1 = change
        else:
            t1 = float("inf")
        time = 0.0
        if current is not None:
            _b[0].rw(current[0])
        speed = 0.0 if current is None else current[1]
        acc = 0.0 if current is None else current[2]
        _b["len"].setgoal(order, value, speed, acc)
        dt = 0.1
        res: list[tuple[float, ...]] = []
        assert len(_b[0].goal)
        res.append((time, _b[0].rw(), _b[0].speed, _b[0].acc))
        while time + dt < t_end:
            if change is not None and abs(time - t1) < dt / 2:
                _b["len"].setgoal(order1, val1)
            _b.step(time, dt)
            time += dt
            res.append((time, *_b[0].current))
        if show:
            do_show(res)
        return res

    boom = np.array((10, 0, 0), float)

    def boom_setter(newval: float | None = None, idx: int = 0):
        if newval is None:
            return boom[idx]
        else:
            boom[idx] = newval
            return newval

    def i_time(res: Sequence[tuple[float, ...]], time: float, eps: float = 0.0001):
        for i in range(len(res)):
            if abs(res[i][0] - time) < eps:
                return i
        raise KeyError(f"Time {time} not found in table")

    _b = Controls(limit_err=logging.CRITICAL)
    _b.append(Control("len", ((-100.0, 90.0), None, (-1.0, 0.5)), partial(boom_setter, idx=0)))
    # fixed velocity and the acceleration limits are not needed:
    _b.append(Control("polar", ((-2.0, 2.0), 1.0), partial(boom_setter, idx=1)))
    _b.append(Control("azimuth", ((-1, 1), (-2, 2), (0, 0)), partial(boom_setter, idx=2)))

    res = do_goal(0, 7.0, current=np.array((1.0, 0.0, 0.0), float), t_end=10.0)

    return
    # set a constant acceleration for 1 time unit
    res = do_goal(2, 0.5, current=np.array((1.0, 0.3, 0.0), float), t_end=1.0)
    assert abs(res[-1][1] - (1.0 + 0.3 * 1 + 0.5 * 0.5 * 1**2)) < 1e-10
    assert abs(res[-1][2] - (0.3 + 0.5 * 1)) < 1e-10
    assert abs(res[-1][3] - 0.5) < 1e-10

    # set a speed goal for 1 time unit
    res = do_goal(1, 3.0, current=np.array((1.0, 0.3, 0.0), float), t_end=10.0)
    assert np.allclose(res[55], (5.5, 10.21, 3.0, 0.0)), f"Goal reached? {res[55]}"
    assert np.allclose(res[-1][1:], (23.71, 3.0, 0.0)), f"Found {res[-1][1:]}"

    # get from one position to another (non-zero velocity on start)
    res = do_goal(0, 0.0, current=np.array((10.0, 1.0, 0.0), float), t_end=3.65)
    assert np.allclose(_b[0].current, (9.375, 0.0, 0.0)), f"Found {_b[0].current}"

    # get from one velocity to another
    res = do_goal(1, -2.0, current=np.array((10.0, 1.0, 0.0), float), t_end=3.2)
    assert np.allclose(_b[0].current, (8.3, -2.0, 0.0)), f"Found {_b[0].current}"

    # get from one position to another (zero velocity and acceleration on both ends)
    res = do_goal(0, 0.0, current=np.array((10.0, 0.0, 0.0), float), t_end=15)
    assert np.allclose(_b[0].current, (-12.5, 0, 0)), f"Found {_b[0].current}"

    # set an acceleration (non-zero position and velocity)
    res = do_goal(2, -0.1, current=np.array((10.0, 1.0, 0.0), float), t_end=2.01)
    expected = (10 + 1.0 * 2.0 - 0.5 * 0.1 * 2.0**2, 1.0 - 0.1 * 2.0, -0.1)
    assert np.allclose(_b[0].current, expected), f"{_b[0].current} != {expected}."

    # set a speed and overwrite it after a time with a new value
    res = do_goal(1, 0.45, current=np.array((10.0, 0, 0), float), t_end=10.0, change=(5.0, 1, 0.0))
    i_acc = i_time(res, 0.45 / 0.5)
    assert np.allclose(res[i_acc], (0.45 / 0.5, 10 + 0.5 * 0.5 * (0.45 / 0.5) ** 2, 0.45, 0.5)), (
        f"After acc.: {res[i_acc]}"
    )
    assert np.allclose(res[-1][1:], (12.14875, 0, 0)), f"Found end state {res[-1]}"

    Controls.limit_err = logging.WARNING  # allow corrections from now on

    # Speed from 5 to 2.2. Detailed analysis
    res = do_goal(1, 2.2, np.array((25.0, 5, 0), float), 10.0)
    tgoal = (2.2 - 5) / (-1.0)
    xgoal = 25.0 + 5 * tgoal + 0.5 * (-1) * tgoal**2
    for t, x, v, a in res:
        if t == 0:
            assert np.allclose((x, v, a), (25.0, 5.0, 0)), f"Found ({x}, {v}, {a})"
        elif t <= tgoal + 1e-13:  # decellerate to reach new velocity
            assert abs(a + 1) < 1e-9, f"@{t} Acc.: {a} != -1"
            assert abs(v - 5.0 - a * t) < 1e-9, f"@{t} Velocity: {v} != {5.0 + a * t}"
            assert abs(x - 25.0 - 5 * t - 0.5 * a * t * t) < 1e-9, f"@{t} Pos.: {x} != {25 + 5 * t - 0.5 * 1 * t * t}"
        else:  # keep new velocity 'forever'
            _x = xgoal + 2.2 * (t - tgoal)
            assert np.allclose((x, v, a), (_x, 2.2, 0.0)), f"@{t}: {(x, v, a)} != {(_x, 2.2, 0)} with {tgoal}, {xgoal}"


if __name__ == "__main__":
    retcode = 0  # pytest.main(["-rA", "-v", "--rootdir", "../", "--show", "False", __file__])
    assert retcode == 0, f"Non-zero return code {retcode}"
    logging.basicConfig(level=logging.DEBUG)
    plt.set_loglevel(level="warning")
    # test_limits()
    test_goal(show=True)
