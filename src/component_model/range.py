import logging
from enum import Enum
from typing import Any, Never, Sequence

from component_model.unit import Unit

logger = logging.getLogger(__name__)


class Range(object):
    """Utility class to store and handle the variable range of a single-valued variable.

    Args:
        val: value for which the range is defined. At least an example value of the same type shall be provided.
        rng (tuple) = (): Optional range of the variable in terms of a tuple of the same type as initial value.
           Should be specified with units (as string).

           * If an empty tuple is specified, the range is automatically determined.
             That is only possible for float or enum type variables, where the former evaluates to (-inf, inf).
             Maximum or minimum int values do not exist in Python, such that these always must be provided explicitly.
             It is not possible to set only one of the elements of the tuple automatically.
           * If None is specified, the initial value is chosen, i.e. no range.
             `None` can be applied to the whole tuple or to single elements of the tuple.
             E.g. (1,None) sets the range to (1, start)
           * For some variable types (e.g. str) no range is expected.
        unit (Unit): expected Unit (should be determined for start value before range is determined)
    """

    def __init__(
        self,
        val: bool | int | float | str | Enum,
        #        rng: tuple[int|float|Enum|str|None,int|float|Enum|str|None]|None|tuple[()] = tuple(), # type: ignore[assignment]
        rng: tuple[Any, Any] | None | Sequence[Never] = tuple(),
        unit: Unit | None = None,
    ):
        self.rng: tuple[int | bool | float | str, int | bool | float | str]
        typ = type(val)
        if unit is None:
            unit = Unit()
        assert isinstance(val, (bool, int, float, str, Enum)), f"Only primitive types allowed for Range. Found {typ}"
        if isinstance(val, str):
            assert unit.u == "dimensionless", "A free string cannot have units."
            self.rng = (val, val)  # no range for free strings
        elif rng is None:  # fixed value in any case
            self.rng = (unit.from_base(val), unit.from_base(val))
        elif isinstance(rng, tuple) and not len(rng):  # empty tuple => try automatic range
            self.rng = Range.auto_extreme(val)  # fails if val is an int variable
        elif (
            isinstance(rng, tuple)
            and len(rng) == 2
            and all(x is None or isinstance(x, (str, int, bool, float, Enum)) for x in rng)
        ):
            l_rng = list(rng)  # work on a mutable object
            for i, r in enumerate(rng):
                if r is None:
                    l_rng[i] = unit.from_base(val)  # replace with fixed value 'val' as display value
                else:
                    assert isinstance(r, (str, int, bool, float, Enum)), f"Found type {type(r)}"
                    check, q = unit.compatible(r, typ, strict=True)
                    if not check:
                        raise ValueError(f"Provided range {rng} is not conformant with unit {unit}") from None
                    q = unit.from_base(q)  # ensure display units
                    assert isinstance(q, (int, bool, float)), "Unexpected type {type(q)} in {rng}[{i}]"
                    try:
                        q = type(val)(q)  # ensure correct Python type
                    except Exception as err:
                        raise TypeError(f"Incompatible types range {rng} - {val}") from err
                    l_rng[i] = q
            self.rng = tuple(l_rng)  # type: ignore  ## cannot see how tuple contains str or None here!
        else:
            raise TypeError(f"Unhandled range specification {rng}) from None")

    @classmethod
    def auto_extreme(cls, var: bool | int | float | str | Enum | type) -> tuple[int | float | bool, int | float | bool]:
        """Return the extreme values of the variable.

        Args:
            var: the variable for which to determine the extremes,
              represented by an instantiated object (example) or by the type itself

        Returns
        -------
            A tuple containing the minimum and maximum value the given variable can have
        """
        if isinstance(var, bool) or (isinstance(var, type) and var is bool):
            return (False, True)
        elif isinstance(var, float) or (isinstance(var, type) and var is float):
            return (float("-inf"), float("inf"))
        elif isinstance(var, Enum) or (isinstance(var, type) and issubclass(var, Enum)):
            if isinstance(var, Enum):
                return (min(x.value for x in type(var)), max(x.value for x in type(var)))
            else:
                return (min(x.value for x in var), max(x.value for x in var))

        else:
            if isinstance(var, type):
                raise ValueError(f"Auto-extremes for type {var} cannot be determined") from None
            else:
                raise ValueError(f"Auto-extremes for type {type(var)} cannot be determined") from None

    def check(
        self,
        value: bool | int | float | str | Enum | None,
        typ: type = float,
        unit: Unit | None = None,
        disp: bool = True,
    ) -> bool:
        """Check a value with respect to type and range.

        Args:
            value: the Python value to check with respect to the internally defined Range
            typ (type): the expected Python type of the value
            unit (Unit): the Unit object related to the variable
            disp (bool): check value as display units (True) or base units (False)
        """
        if unit is None:
            unit = Unit()
        if value is None:  # denotes unchanged values (of compound variables)
            return True
        if not isinstance(value, typ):
            try:
                value = typ(value)  # try to cast the values
            except Exception:  # give up
                return False
        # special types
        if typ is str:  # no range checking on str
            return True
        elif typ is bool:
            return isinstance(value, bool)
        elif isinstance(value, Enum):
            return isinstance(value, typ)

        elif isinstance(value, (int, float)) and all(isinstance(x, (int, float)) for x in self.rng):
            assert typ is int or typ is float, f"Inconsistent type {typ}. Expect int or float"
            if not disp and unit.du is not None:  # check an internal unit values
                value = unit.from_base(value)
            return self.rng[0] <= value <= self.rng[1]  # type: ignore[operator] ## There is no str involved!
        else:
            logger.error(f"range check(): value={value}, type={typ}, range={self.rng}")
            return False

    @classmethod
    def is_valid_spec(
        cls, rng: tuple[Any, ...] | tuple[Any, Any] | None | Sequence[Never], var_len: int, typ: type, level: int = 0
    ) -> int:
        """Check whether the supplied rng is a valid range specification for a variable.
        Applies to scalar and compound variable specs.
        Return 0 (ok) or error code >0 if not ok.
        """
        if rng is None:
            ck = 0  # fixed value(s)
        elif isinstance(rng, tuple) and not len(rng):  # all automatic
            ck = int(typ is int)  # 1/0 (not possible for int)
        elif isinstance(rng, tuple):  # need a tuple now
            if var_len == 1:
                if len(rng) != 2:  # scalar specified by a 2-tuple
                    ck = 2
                else:  # final check of scalar spec 2-tuple
                    ck = 0
                    for i, r in enumerate(rng):
                        if r is not None and not isinstance(r, (int, bool, float, Enum, str)):
                            ck += 10 + i
                    if not any(isinstance(rng[i], str) for i in range(2)):
                        if rng[0] > rng[1]:  # wrong order
                            ck += 10 + 9

            elif var_len > 1:
                if len(rng) != var_len:  # one range for each variable
                    ck = 3
                else:
                    ck = 0
                    for i, r in enumerate(rng):
                        ck += Range.is_valid_spec(r, 1, typ, level=(i + 1) * 100)
        else:
            ck = 4  # would need a tuple here
        return ck if ck == 0 else level + ck
