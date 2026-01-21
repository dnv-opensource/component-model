import logging
from enum import Enum
from functools import partial
from typing import Callable

import numpy as np
from pint import Quantity, UnitRegistry  # management of units

logger = logging.getLogger(__name__)


class Unit:
    """Helper class to store and manage units and display units,
    i.e. base unit of variable and unit differences 'outside' and 'inside' the model.

    One Unit object represents one scalar variable.
    """

    _ureg: UnitRegistry | None = None

    def __init__(self, quantity: bool | int | float | str | Enum | None = None, typ: type | None = None):
        assert Unit._ureg is not None, "Before units can be instantiated, Unit.ensure_unit_registry() must be called."
        # properties with default values. Initialized through parse_quantity
        self.u: str = "dimensionless"  # default: dimensionless unit (placeholder)
        self.du: str | None = None  # display unit (default: same as u, no transformation)
        self.to_base: Callable[float] = partial(Unit.identity)  # f(display-value) -> base-value
        self.from_base: Callable[float] = partial(Unit.identity)  # f(base-value) -> display-value
        if quantity is not None:  # if parse-value is called on class it also returns the (parsed,converted) base-value
            _val = self.parse_quantity(quantity, typ)

    @classmethod
    def ensure_unit_registry(cls, system: str = "SI", autoconvert: bool = True):
        cls._ureg = UnitRegistry(system=system, autoconvert_offset_to_baseunit=autoconvert)
        return cls._ureg

    def __str__(self):
        txt = f"Unit {self.u}, display:{self.du}"
        if self.du is not None:
            txt += f". Offset:{self.to_base(0)}, factor:{self.to_base(1.0) - self.to_base(0.0)}"
        return txt

    def parse_quantity(
        self, quantity: bool | int | float | str | Enum, typ: type | None = None
    ) -> bool | int | float | str | Enum:
        """Parse the provided quantity in terms of magnitude and unit, if provided as string.
        If another type is provided, dimensionless units are assumed.

        Args:
            quantity: the quantity to disect. Should be provided as string, but also the trivial cases (int,float,Enum) are allowed.
            A free string should not be used and leads to a warning
        Returns:
            the magnitude in base units, the base unit and the unit as given (display units),
            together with the conversion functions between the units.
        """
        if typ is str:
            self.u = "dimensionless"
            self.du = None
            val = quantity
        elif isinstance(quantity, str):  # only string variable make sense to disect
            try:
                q = Unit._ureg(quantity)  # parse the quantity-unit and return a Pint Quantity object
                if isinstance(q, (int, float)):
                    self.u = ""
                    self.du = None
                    return q  # integer or float variable with no units provided
                elif isinstance(q, Quantity):  # pint.Quantity object
                    # transform to base units ('SI' units). All internal calculations will be performed with these
                    val = self.val_unit_display(q)
                else:
                    logger.critical(f"Unknown quantity {quantity} to disect")
                    raise ValueError(f"Unknown quantity {quantity} to disect") from None
            # no recognized units. Assume a free string. ??Maybe we should be more selective about the exact error type:
            except Exception as warn:
                logger.warning(f"Unhandled quantity {quantity}: {warn}. A str? Set explicit 'typ=str'.")
                self.u = ""
                self.du = None
                val = str(quantity)
        else:
            self.u = "dimensionless"
            self.du = None
            val = quantity
        if typ is not None and type(val) is not typ:  # check variable type
            try:  # try to convert the magnitude to the correct type.
                val = typ(val)
            except Exception as err:
                logger.critical(f"Value {val} is not of the correct type {typ}")
                raise TypeError(f"Value {val} is not of the correct type {typ}") from err
        return val

    @classmethod
    def linear(cls, x: float, b: float, a: float = 0.0):
        return a + b * x

    @classmethod
    def identity(cls, x: float):
        return x

    def val_unit_display(self, q: Quantity[float]) -> float:
        """Identify base units and calculate the transformations between display and base units.

        Returns
        -------
            The numerical value of q. As side effect

            * the unit `u` is set. Might be `dimensionless`
            * the display unit `du` is set to None if same as unit, else

               - it is set to the display unit name and
               - the transformations `to_base` and `from_base` are set.
        """
        assert Unit._ureg is not None, "Need UnitRegistry at this point"
        qb = q.to_base_units()
        self.u = str(qb.units)
        val = qb.magnitude  # Note: numeric types are not converted, e.g. int to float
        if qb.units == q.units:  # no conversion
            self.du = None
        else:  # calculate the conversion functions
            # we generate a second value and calculate the straight line conversion function
            # did not find a better way in pint
            self.du = str(q.units)
            q2 = Unit._ureg.Quantity(10.0 * (q.magnitude + 10.0), q.units)
            qb2 = q2.to_base_units()
            a = (qb.magnitude * q2.magnitude - qb2.magnitude * q.magnitude) / (q2.magnitude - q.magnitude)
            b = (qb2.magnitude - qb.magnitude) / (q2.magnitude - q.magnitude)
            if abs(a) < 1e-9:  # multiplicative conversion
                if abs(b - 1.0) < 1e-9:  # unit and display unit are compatible. No transformation
                    self.du = None
                self.to_base = partial(Unit.linear, b=b)
                self.from_base = partial(Unit.linear, b=1.0 / b)
            else:  # there is a constant (e.g. Celsius to Fahrenheit)
                self.to_base = partial(Unit.linear, b=b, a=a)
                self.from_base = partial(Unit.linear, b=1.0 / b, a=-a / b)
        return val

    @classmethod
    def make(
        cls, quantity: bool | int | float | str | Enum, typ: type | None = None
    ) -> tuple[tuple[bool | int | float | str | Enum], tuple["Unit"]]:
        """Parse quantity and return the resulting value and its unit object."""
        u = Unit()
        val = u.parse_quantity(quantity, typ)
        return ((val,), (u,))

    @classmethod
    def make_tuple(
        cls,
        quantities: tuple[bool | int | float | str | Enum, ...] | list[bool | int | float | str | Enum] | np.ndarray,
        typ: type | None = None,
    ) -> tuple[tuple[bool | int | float | str | Enum, ...], tuple["Unit", ...]]:
        """Make a tuple of values and Unit objects from the tuple of quantities, using make()."""
        values: list[bool | int | float | str | Enum] = []
        units: list[Unit] = []
        for q in quantities:
            val, u = cls.make(q, typ)
            values.extend(val)
            units.extend(u)
        return (tuple(values), tuple(units))

    @classmethod
    def derivative(cls, baseunits: tuple["Unit", ...], tu: str = "s") -> tuple[tuple[float, ...], tuple["Unit", ...]]:
        """Construct units for a derivative variable of basevars. tu is the time unit."""
        units: list["Unit"] = []
        for bu in baseunits:
            u = Unit()
            u.u = f"{bu.u}/{tu}"
            u.du = None if bu.du is None else f"{bu.du}/{tu}"
            if bu.du is not None:
                u.to_base = bu.to_base
                u.from_base = bu.from_base
            units.append(u)
        values = [0.0] * len(baseunits)
        return (tuple(values), tuple(units))

    def compatible(
        self, quantity: bool | int | float | str | Enum, typ: type | None = None, strict: bool = True
    ) -> tuple[bool, bool | int | float | str | Enum]:
        """Check whether the supplied quantity 'q' is compatible with this unit.
        If strict==True, the supplied quantity shall be in display units.
        """
        _q, _unit = Unit.make(quantity, typ)
        q = _q[0]
        unit = _unit[0]
        # no explicit unit needed when the quantity is 0 or inf (anything compatible)
        if (
            (
                (q == 0 or q == float("inf") or q == float("-inf")) and unit.u == "dimensionless"
            )  # 0, +/-inf without unit
            or (strict and self.u == unit.u and self.du == unit.du)
            or (not strict and self.u == unit.u)
        ):
            return (True, q)
        else:
            return (False, q)
