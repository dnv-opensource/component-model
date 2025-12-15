import logging

import numpy as np

logger = logging.getLogger(__name__)


class Controls(object):
    """Keep track of float variable changes.

    * Store and check possible float variable changes, including first and second derivatives
    * Set control goals. A goal is either None or a sequence of (time, acceleration) tuples.
      In this way an acceleration can be set or the velocity or position can be changed through the step function.

    Args:
        names (tuple[str]): Tuple of name strings for the control variables to use.
          The names are only used internally an do not need to correlate with outside names and objects
        limits: None or tuple of limits one per name. None means 'no limit' for variable(s), order or min/max.
          In general the (min,max) is provided for all orders, i.e. 3 tuples of 2-tuples of float per name.
          A given order can be fixed through min==max or by providing a single float instead of the tuple.
          The sub-orders of a fixed order do not need to be provided and are internally set to (0.0, 0,0)
        limit_err: Determines how limit errors are dealt with.
          Anything below critical sets the value to the limit and provides a logger message.
          Critical leads to a program run error.
    """

    def __init__(
        self,
        names: tuple[str, ...] = (),
        limits: tuple[tuple[tuple[float | None, float | None] | float | None, ...], ...] | None = None,
        limit_err: int = logging.WARNING,
    ):
        self.names = list(names)
        self.dim = len(names)
        self.goals: list[tuple | None] = [] if self.dim == 0 else [None] * self.dim  # Initially no goals set
        self.rows = [] if self.dim == 0 else [-1] * self.dim  # current row in goal sequence
        self.current: list[np.ndarray] = (
            [] if self.dim == 0 else [np.array((0.0, 0.0, 0.0), float) * self.dim]
        )  # current positions, speeds, accelerations
        self.nogoals: bool = True
        self.limit_err = limit_err
        self._limits: list[list[tuple[float, float]]] = []
        if isinstance(limits, tuple):
            for idx in range(self.dim):
                self._limits.append(self._prepare_limits(limits[idx]))

    def _prepare_limits(
        self, limits: tuple[tuple[float | None, float | None] | float | None, ...]
    ) -> list[tuple[float, float]]:
        """Prepare and check 'limits', so that they can be appended to Controls.

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

    def append(self, name: str, limits: tuple[tuple[float | None, float | None] | float | None, ...]):
        self.names.append(name)
        self.dim += 1
        self.goals.append(None)
        self.rows.append(-1)
        self.current.append(np.array((0.0, 0.0, 0.0), float))
        self._limits.append(self._prepare_limits(limits))

    def idx(self, name: str) -> int:
        """Find index from name."""
        return self.names.index(name)

    def limit(self, ident: int | str, order: int, minmax: int, value: float | None = None) -> float:
        """Get/Set the single limit for 'idx', 'order', 'minmax'."""
        idx = ident if isinstance(ident, int) else self.names.index(ident)
        assert 0 <= order < 3, f"Only order = 0,1,2 allowed. Found {order}"
        assert 0 <= minmax < 2, f"Only minmax = 0,1 allowed. Found {minmax}"
        if value is not None:
            lim = self._limits[idx][order]
            self.limits(idx, order, (value if minmax == 0 else lim[0], value if minmax == 1 else lim[1]))
        return self._limits[idx][order][minmax]

    def limits(self, ident: int | str, order: int, value: tuple | None = None) -> tuple[float, float]:
        """Get/Set the min/max limit for 'idx', 'order'."""
        idx = ident if isinstance(ident, int) else self.names.index(ident)
        assert 0 <= idx < 3, f"Only idx = 0,1,2 allowed. Found {idx}"
        assert 0 <= order < 3, f"Only order = 0,1,2 allowed. Found {order}"
        if value is not None:
            assert value[0] <= value[1], f"Wrong order:{value}"
            self._limits[idx][order] = value
        return self._limits[idx][order]

    def check_limit(self, ident: int | str, order: int, value: float) -> float | None:
        idx = ident if isinstance(ident, int) else self.names.index(ident)
        if value < self.limit(idx, order, 0):  # check goal value wrt. limits
            msg = f"Goal value {value} is below the limit {self.limit(idx, order, 0)}."
            if self.limit_err == logging.CRITICAL:
                raise ValueError(msg + "Stopping execution.") from None
            else:
                logger.log(self.limit_err, msg + " Setting value to minimum.")
                return self.limit(idx, order, 0)  # corrected value
        if value > self.limit(idx, order, 1):
            msg = f"Goal value {value} is above the limit {self.limit(idx, order, 1)}."
            if self.limit_err == logging.CRITICAL:
                raise ValueError(msg + "Stopping execution.") from None
            else:
                logger.log(self.limit_err, msg + " Setting value to maximum.")
                return self.limit(idx, order, 1)  # corrected value
        return value

    def setgoal(self, ident: int | str, order: int, value: float | None, t0: float = 0.0):
        """Set a new goal for 'ident', i.e. set the required time-acceleration sequence
        to reach value with all derivatives = 0.0.

        Args:
            ident (int|str): the identificator of the control element (as integer or name)
            order (int): the order 0,1,2 of the goal to be set
            value (float|None): the goal value (acceleration, velocity or position) to be reached.
              None to unset the goal.
            t0 (float): the current time
        """
        idx = ident if isinstance(ident, int) else self.names.index(ident)
        # check the index, the order and the value with respect to limits
        if not 0 <= order < 3:
            raise ValueError(f"Only order = 0,1,2 allowed. Found {order}") from None
        # assert value is None or self.goals[idx] is None, "Change of goals is currently not implemented."
        if value is None:  # unset goal
            self.goals[idx] = None
        else:
            value = self.check_limit(idx, order, value)
            # print(f"SET {idx}, {order}: {value}. Current:{current}. Limits:{self.limits(idx, order)}")
            if (
                (
                    order == 0 and abs(self.current[idx][0] - value) < 1e-13
                )  # (adjusted) position goal is already reached
                or (order == 2 and value == 0.0)
            ):  # zero acceleration requested
                self.goals[idx] = None
            elif order == 2:  # set the acceleration from now and 'forever'
                self.goals[idx] = ((float("inf"), value),)
            elif order == 1:  # accelerate to another velocity and keep that 'forever'
                _speed = self.current[idx][1]
                acc = self.limit(idx, 2, int(_speed < value))  # maximum acceleration or deceleration
                self.goals[idx] = ((t0 + (value - _speed) / acc, acc), (float("inf"), 0.0))
            elif order == 0:  # sequence of acceleration and deceleration to reach a new position
                _pos = self.current[idx][0]
                if abs(self.current[idx][1]) > 1e-12:  # the initial velocity is not zero. Need to decelerate
                    v0 = self.current[idx][1]
                    a = self.limit(idx, 2, int(bool(-np.sign(v0) + 1)))
                    goal0 = (t0, a)
                    t0 += -v0 / a
                    _pos = -v0 * v0 / 2 / a  # updated position when the velocity is zero
                else:
                    goal0 = None
                acc1 = self.limit(idx, 2, int(_pos < value))  # maximum acceleration on first leg
                acc2 = self.limit(idx, 2, int(_pos > value))  # maximum acceleration on last leg
                if acc1 == 0 or acc2 == 0:
                    _acc = np.sign(int(_pos < value) + 1) * float("inf")
                else:
                    _acc = 0.5 * (1.0 / acc1 - 1.0 / acc2)
                vmax = self.limit(idx, 1, int(_pos < value))  # maximum velocity towards goal
                dx1_dx3 = vmax**2 * _acc
                dx2 = value - _pos - dx1_dx3
                if np.sign(value - _pos) != np.sign(dx2):  # maximum velocity is not reached
                    v1 = np.sign(value - _pos) * np.sqrt(_acc * (value - _pos))
                    dt1 = v1 / acc1
                    dt2 = -v1 / acc2
                    if goal0 is None:
                        self.goals[idx] = ((t0 + dt1, acc1), (t0 + dt1 + dt2, acc2), (float("inf"), 0.0))
                    else:
                        self.goals[idx] = (goal0, (t0 + dt1, acc1), (t0 + dt1 + dt2, acc2), (float("inf"), 0.0))
                else:
                    dt1 = vmax / acc1
                    dt2 = dx2 / vmax
                    dt3 = -vmax / acc2
                    if goal0 is None:
                        self.goals[idx] = (
                            (t0 + dt1, acc1),
                            (t0 + dt1 + dt2, 0.0),
                            (t0 + dt1 + dt2 + dt3, acc2),
                            (float("inf"), 0.0),
                        )
                    else:
                        self.goals[idx] = (
                            goal0,
                            (t0 + dt1, acc1),
                            (t0 + dt1 + dt2, 0.0),
                            (t0 + dt1 + dt2 + dt3, acc2),
                            (float("inf"), 0.0),
                        )
        logger.info(f"New goals: {self.goals}")
        self.nogoals = all(x is None for x in self.goals)
        self.rows[idx] = -1 if self.goals[idx] is None else 0  # set start row

    def getgoal(self, ident: int | str) -> tuple:
        idx = ident if isinstance(ident, int) else self.names.index(ident)
        assert 0 <= idx < 3, f"Only idx = 0,1,2 allowed. Found {idx}"
        return (self.current[idx], self.goals[idx])

    def step(self, time: float, dt: float):
        """Step towards the goals (if goals are set)."""
        if not self.nogoals:
            for idx in range(self.dim):
                goals = self.goals[idx]
                if goals is not None:
                    _time = time # copies needed in case that there are several goals
                    _dt = dt
                    _current = self.current[idx]

                    _t, _current[2] = goals[self.rows[idx]]
                    while _time > _t:  # move row so that it starts in the right time-acc row
                        self.rows[idx] += 1
                        _t, _current[2] = goals[self.rows[idx]]
                    while _dt > 0:
                        if _time + _dt < _t:  # covers the whole
                            _current[0] = self.check_limit(
                                idx, 0, _current[0] + _current[1] * _dt + 0.5 * _current[2] * _dt * _dt
                            )
                            _current[1] = self.check_limit(idx, 1, _current[1] + _current[2] * _dt)
                            _dt = 0
                        else:  # dt must be split
                            dt1 = _t - _time
                            _current[0] = self.check_limit(
                                idx, 0, _current[0] + _current[1] * dt1 + 0.5 * _current[2] * dt1 * dt1
                            )
                            _current[1] = self.check_limit(idx, 1, _current[1] + _current[2] * dt1)
                            _time = _t
                            _dt -= dt1
                            self.rows[idx] += 1
                            _t, _current[2] = goals[self.rows[idx]]

                    if np.isinf(_t) and abs(_current[2]) < 1e-12 and abs(_current[1]) < 1e-12:
                        self.goals[idx] = None
        self.nogoals = all(x is None for x in self.goals)
