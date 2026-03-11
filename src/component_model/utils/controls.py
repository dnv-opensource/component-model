import logging
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)


class RW(Protocol):
    """Defines the read/write access function for a single controll, i.e. the .rw property.

    Since it is desired to have a default empty argument list (for read access),
    a Protocol must be used for proper type checking. Implement as

    Implement either through a new class, implementing the full __call__ method or use functool.partial().
    See `tests/test_controls.py` for examples
    """

    def __call__(self, val: float | None = None, /) -> float: ...


class Control(object):
    """Keep track of the changes of a single float variable, avoiding discontinuities and abrupt changes.

    Based on the ranges of value, speed, acceleration, the following rules are used:

    #. Instantaneous acceleration changes (within the limits set for acceleration) are allowed
    #. When a goal is reached, the goal value (variable value, first or second derivation) is kept 'forever'
    #. Derivatives above 2nd order are not considered.

    The goal shall be reached in as short time as possible.

    * Store and check possible float variable changes, including first and second derivatives
    * Set control goals as list of goals, one goal per control variable.
    * A single goal is either None (currently inactive) or a tuple of (time, acceleration)-tuples.
      In this way an acceleration can be set or the velocity or position can be changed through the step function.

    Args:
        name (str): Name of control variable. Unique within Controls.
          The names are only used internally an do not need to correlate with outside names and objects
        limits: None or tuple of limits one per name. None means 'no limit' for variable(s), order or min/max.
          In general the (min,max) is provided for all orders, i.e. 3 tuples of 2-tuples of float per name.
          A given order can be fixed through min==max or by providing a single float instead of the tuple.
          The sub-orders of a fixed order do not need to be provided and are internally set to (0.0, 0,0)
        rw (RW): getter/setter function for the variable related to the control.
          The function shall have a single optional (float or None) argument and return the value of the variable.
          If omittet, or None, only the current value is returned. If not None, the new value is set and returned.
    """

    def __init__(
        self,
        name: str,
        limits: tuple[tuple[float | None, float | None] | float | None, ...],
        rw: RW,
    ):
        self.name = name
        self._limits = Control._prepare_limits(limits)
        self.rw = rw
        self.started: bool = False
        self.goal: list[tuple[float, float]] = []
        self.speed: float = 0.0  # may be used during goal tracking
        self.acc: float = 0.0  # may be used during goal tracking

    @staticmethod
    def _prepare_limits(
        limits: tuple[tuple[float | None, float | None] | float | None, ...],
    ) -> list[tuple[float, float]]:
        """Prepare and check 'limits', so that they can be stored in Control.

        Args:
            limits : optional specification of limits for var, d_var/dt and d2_var/dt2 of single float variable:

            * None: Denotes 'no specified limits' => -inf ...inf
            * single number: The variable is fixed to this single value
            * tuple(min,max): minimum and maximum value
        """
        _limits = [(float("-inf"), float("inf"))] * 3  # default limits for value, 1.deriv., 2.deriv

        if limits is None:  # default values for all orders
            return _limits

        for order in range(3):
            if len(limits) <= order:
                if _limits[order - 1][0] == _limits[order - 1][1]:  # order-1 fixed
                    _limits[order] = (0.0, 0.0)  # derivative fixed zero
                else:
                    raise ValueError(f"Explicit limit needed for order {order} in {limits}.") from None
            else:
                lim = limits[order]
                if lim is None:  # use the default value
                    pass
                elif isinstance(lim, (float, int)):  # single value provided
                    assert lim
                    fixed = float(lim)
                    _limits[order] = (fixed, fixed)
                elif isinstance(lim, tuple):  # both values provided
                    assert len(lim) == 2, f"Need both minimum and maximum. Found {limits[order]}"
                    if lim[0] is not None and lim[1] is not None:
                        _limits[order] = (lim[0], lim[1])
                        assert _limits[order][0] <= _limits[order][1], f"Wrong order of limits: {limits[order]}"
                    elif lim[0] is not None:
                        _limits[order] = (float(lim[0]), float("inf"))
                    elif lim[1] is not None:
                        _limits[order] = (float("-inf"), float(lim[1]))
                else:
                    raise ValueError(f"Unknown type of limits[{order}]: {lim}") from None
        return _limits

    def limit(self, order: int, minmax: int, value: float | None = None) -> float:
        """Get/Set the limit for the Control, 'order', 'minmax'."""
        assert 0 <= order < 3, f"Only order = 0,1,2 allowed. Found {order}"
        assert 0 <= minmax < 2, f"Only minmax = 0,1 allowed. Found {minmax}"
        if value is not None:
            lim = self._limits[order]
            self.limits(order, (value if minmax == 0 else lim[0], value if minmax == 1 else lim[1]))
        return self._limits[order][minmax]

    def limits(self, order: int, value: tuple[float, float] | None = None) -> tuple[float, float]:
        """Get/Set the min/max limit for 'idx', 'order'."""
        assert 0 <= order < 3, f"Only order = 0,1,2 allowed. Found {order}"
        if value is not None:
            assert value[0] <= value[1], f"Wrong order:{value}"
            self._limits[order] = value
        return self._limits[order]

    def check_limit(self, order: int, value: float) -> float:
        for k in range(2):
            lim = self.limit(order, k)
            err = (lim - value) if k == 0 else (value - lim)
            if err > 0:  # goal exceeded
                if err > 1e-13:  # not a minor (probably numerical) issue. Message or error
                    side = "below" if k == 0 else "above"
                    msg = f"Goal '{self.name}'@ {value} is {side} the limit {lim}."
                    if Controls.limit_err == logging.CRITICAL:
                        raise ValueError(msg + " Stopping execution.") from None
                    else:
                        logger.log(Controls.limit_err, msg + " Setting value to minimum.")
                return lim
        return value

    def setgoal(self, order: int, value: float | None, speed: float | None = None, acc: float | None = None):
        """Set a new goal for the control, i.e. set the required time-acceleration sequence
        to reach value with all derivatives = 0.0.

        .. note:: Initially the start-time for the goal sequence is unknown and is set to zero.
          At the first step the current time is added to these times

        Args:
            order (int): the order 0,1,2 of the goal to be set
            value (float|None): the goal value (acceleration, velocity or position) to be reached.
              None to unset the goal.
            speed (float): Optional possibility to set a start-speed != 0.0.
              None: keep the internally stored value
            acc (float): Optional possibility to set a start-acceleration != 0.0.
              None: keep the internally stored value
        """
        # check the order and the value with respect to limits
        if not 0 <= order < 3:
            raise ValueError(f"Only order = 0,1,2 allowed. Found {order}") from None
        if speed is not None:
            self.speed = speed
        if acc is not None:
            self.acc = acc
        if value is None:  # unset goal
            self.goal = []
        else:
            value = self.check_limit(order, value)
            assert value is not None, "float value expected here"
            # print(f"SET {order}: {value}. Current:{current}. Limits:{self.limits(order)}")
            if (
                (order == 0 and abs(self.rw() - value) < 1e-13)  # position goal already reached
                or (order == 1 and abs(self.speed - value) < 1e-13)  # speed goal already reached
                or (order == 2 and abs(self.acc - value) < 1e-13)
            ):  # nothing to do
                self.goal = []
            elif order == 2:  # set the acceleration from now and 'forever'
                self.goal = [(float("inf"), value)]
            elif order == 1:  # accelerate to another velocity and keep that 'forever'
                acc = self.limit(2, int(self.speed < value))  # maximum acceleration or deceleration
                self.goal = [((value - self.speed) / acc, acc), (float("inf"), 0.0)]
            elif order == 0:  # sequence of acceleration and deceleration to reach a new position
                _pos = self.rw()
                t0 = 0.0
                if abs(self.speed) > 1e-12:  # the initial velocity is not zero. Need to decelerate
                    v0 = self.speed
                    a = self.limit(2, int(bool(-np.sign(v0) + 1)))
                    goal0 = (0, a)
                    t0 += -v0 / a
                    _pos = -v0 * v0 / 2 / a  # updated position when the velocity is zero
                else:
                    goal0 = None  # start from zero velocity
                acc1 = self.limit(2, int(_pos < value))  # maximum acceleration on first leg
                acc2 = self.limit(2, int(_pos > value))  # maximum acceleration on last leg
                if acc1 == 0 or acc2 == 0:
                    _acc = np.sign(int(_pos < value) + 1) * float("inf")
                else:
                    _acc = 0.5 * (1.0 / acc1 - 1.0 / acc2)
                vmax = self.limit(1, int(_pos < value))  # maximum velocity towards goal
                dx1_dx3 = vmax**2 * _acc
                dx2 = value - _pos - dx1_dx3
                if np.sign(value - _pos) != np.sign(dx2):  # maximum velocity is not reached
                    v1 = np.sign(value - _pos) * np.sqrt(_acc * (value - _pos))
                    dt1 = v1 / acc1
                    dt2 = -v1 / acc2
                    if goal0 is None:
                        self.goal = [(t0 + dt1, acc1), (t0 + dt1 + dt2, acc2), (float("inf"), 0.0)]
                    else:
                        self.goal = [
                            goal0,
                            (t0 + dt1, acc1),
                            (t0 + dt1 + dt2, acc2),
                            (float("inf"), 0.0),
                        ]
                else:
                    dt1 = vmax / acc1
                    dt2 = dx2 / vmax
                    dt3 = -vmax / acc2
                    if goal0 is None:
                        self.goal = [
                            (t0 + dt1, acc1),
                            (t0 + dt1 + dt2, 0.0),
                            (t0 + dt1 + dt2 + dt3, acc2),
                            (float("inf"), 0.0),
                        ]
                    else:
                        self.goal = [
                            goal0,
                            (t0 + dt1, acc1),
                            (t0 + dt1 + dt2, 0.0),
                            (t0 + dt1 + dt2 + dt3, acc2),
                            (float("inf"), 0.0),
                        ]
        self.started = False

    @property
    def current(self):
        """Return the tuple of current value, speed and acceleration."""
        return (self.rw(), self.speed, self.acc)

    def step(self, time: float, dt: float):
        """Step towards the goal (if goal is set)."""
        if len(self.goal):
            if not self.started:  # not yet started. Need to add current time.
                for i, (t, a) in enumerate(self.goal):
                    self.goal[i] = (t + time, a)
                logger.info(f"@{time}. New goal({self.name}): {self.goal}")
                self.started = True
            _t, self.acc = self.goal[0]
            if time > _t:  # move to correct goal entry
                self.goal.pop(0)
                _t, self.acc = self.goal[0]
            while dt > 0:
                _dt = dt if time + dt < _t else _t - time  # may need to split dt if sub-goal ends within
                self.rw(self.check_limit(0, self.rw() + self.speed * _dt + 0.5 * self.acc * _dt * _dt))
                self.speed = self.check_limit(1, self.speed + self.acc * _dt)
                dt -= _dt
                if abs(dt) < 1e-13:
                    break
                else:  # start with the next sub-goal
                    time += _dt
                    self.goal.pop(0)
                    _t, self.acc = self.goal[0]

            if np.isinf(_t) and abs(self.acc) < 1e-12 and abs(self.speed) < 1e-12:
                logger.info(f"@{time}. Goal {self.name} finalized.")
                self.goal = []


class Controls(object):
    """Keep track of float variable changes.

    limit_err: Determines how limit errors are dealt with.
      Anything below critical sets the value to the limit and provides a logger message.
      Critical leads to a program run error.
    """

    limit_err: int = logging.WARNING

    def __init__(
        self,
        limit_err: int = logging.WARNING,
    ):
        Controls.limit_err = limit_err
        self.controls: list[Control] = []

    @property
    def nogoals(self):
        for crl in self.controls:
            if len(crl.goal):
                return False
        return True

    def append(self, crl: "Control"):
        """Append one or several Control object(s)."""
        for c in self.controls:
            if c.name == crl.name:
                raise KeyError(f"Control with name {c.name} already exists. Choose a unique name.") from None
        self.controls.append(crl)

    def extend(self, crls: tuple[Control, ...]):
        for crl in crls:
            self.append(crl)

    def __getitem__(self, ident: int | str):
        """Get the control object identified by ident (index within .controls or valid name)."""
        if isinstance(ident, str):
            for crl in self.controls:
                if crl.name == ident:
                    return crl
            raise KeyError(f"Control with name {ident} not found.") from None
        elif isinstance(ident, int):
            if ident < 0 or ident >= len(self.controls):
                raise ValueError(f"Control {ident} does not exist within set of controls.") from None
            return self.controls[ident]
        else:
            raise TypeError(f"Integer expected as subscript. Found {ident}") from None

    def step(self, time: float, dt: float):
        """Step towards the goals (if goals are set)."""
        if not self.nogoals:
            for crl in self.controls:
                crl.step(time, dt)
