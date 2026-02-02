import logging
from enum import Enum
from functools import partial
from typing import Any, Callable

import numpy as np
from pint import Quantity, UnitRegistry  # management of units

logger = logging.getLogger(__name__)


class Unit:
    """Helper class to store and manage units and display units,
    i.e. base unit of variable and unit differences 'outside'(display units) and 'inside'(base units) the model.

    Args:
        quantity (bool, int, float, str, Enum, None): The quantity to be disected for unit definition. 3 possibilites:

        * None: no units. Instantiates an 'empty' Unit object
        * str: is parsed to disect the unit.
          If a unit is identified this is treated as a variable with units (and possibly display units)
          If no unit is identified this is treated as a free string variable (no units). Use 'None' to ensure free str.
        * bool, int, Enum: variables with no units and no display units

    * one Unit object represents one scalar variable.
    * many variables do not have units (i.e. str, Enum, int  variables). If These get the unit .u=""
    * variables without separate display units get the display unit .du=None and a reduced set of properties
    * only float variables may have separate display units and transformations
    """

    _ureg: UnitRegistry[Any] | None = None

    def __init__(self, quantity: bool | int | float | str | Enum | None = None):
        assert Unit._ureg is not None, "Before units can be instantiated, Unit.ensure_unit_registry() must be called."
        self.u: str = ""  # default: no units
        self.du: str | None = None  # display unit (default: same as u, no transformation)
        self.to_base: Callable[[Any], Any] = Unit.identity  # default transformation is identity
        self.from_base: Callable[[Any], Any] = Unit.identity  # default transformation is identity
        if quantity is not None:  # if parse-value is called on class it also returns the (parsed,converted) base-value
            _val = self.parse_quantity(quantity)

    @classmethod
    def ensure_unit_registry(cls, system: str = "SI", autoconvert: bool = True):
        cls._ureg = UnitRegistry(system=system, autoconvert_offset_to_baseunit=autoconvert)
        return cls._ureg

    def __str__(self):
        txt = f"Unit {self.u}, display:{self.du}"
        if self.du is not None and not (self.to_base is None or self.from_base is None):
            txt += f". Offset:{self.to_base(0)}, factor:{self.to_base(1.0) - self.to_base(0.0)}"
        return txt

    def parse_quantity(self, quantity: bool | int | float | str | Enum) -> bool | int | float | str | Enum:
        """Parse the provided quantity in terms of magnitude and unit, if provided as string.
        If another type is provided, no units are assumed.

        Args:
            quantity: the quantity to disect. Should be provided as string, but also the trivial cases (int,float,Enum) are allowed.
            A free string should not be used and leads to a warning
        Returns:
            the magnitude in base units, the base unit and the unit as given (display units),
            together with the conversion functions between the units.
        """
        if isinstance(quantity, str):  # only string variable make sense to disect
            assert Unit._ureg is not None, "UnitRegistry not yet instantiated!"
            try:
                q = Unit._ureg(quantity)  # parse the quantity-unit and return a Pint Quantity object
                if isinstance(q, (int, float)):
                    self.u = "dimensionless"
                    self.du = None
                    return q  # integer or float variable with no units provided
                elif isinstance(q, Quantity):  # pint.Quantity object
                    # transform to base units ('SI' units). All internal calculations will be performed with these
                    val = self.val_unit_display(q)
                    return val
                else:  # since this is not a recognized quantity, we assume an implicit str
                    pass
            except Exception as err:
                logger.warning(f"Quantity {quantity} could not be disected: {err}. Assume free string.")
        self.u = ""
        self.du = None
        return quantity

    @classmethod
    def identity(cls, val: Any) -> Any:
        return val

    @classmethod
    def slope(cls, val: float, slope: float) -> float:
        return slope * val

    @classmethod
    def linear(cls, val: float, intercept: float, slope: float) -> float:
        return intercept + slope * val

    def val_unit_display(self, q: Quantity[int | float]) -> int | float:
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
            if abs(a) < 1e-9 and abs(b) < 1e-9:  # identity
                self.to_base = self.from_base = Unit.identity
            if abs(a) < 1e-9:  # multiplicative conversion (only slope)
                self.to_base = partial(Unit.slope, slope=b)
                self.from_base = partial(Unit.slope, slope=1.0 / b)
            else:  # there is a constant (e.g. Celsius to Fahrenheit)
                self.to_base = partial(Unit.linear, intercept=a, slope=b)
                self.from_base = partial(Unit.linear, intercept=-a / b, slope=1.0 / b)
        return val

    @classmethod
    def make(
        cls, quantity: bool | int | float | str | Enum, no_unit: bool = False
    ) -> tuple[tuple[bool | int | float | str | Enum], tuple["Unit"]]:
        """Parse quantity and return the resulting value and its unit object.
        If no_unit, only a default object is generated.
        """
        u = Unit()
        if no_unit:
            return ((quantity,), (u,))
        else:
            val = u.parse_quantity(quantity)
            return ((val,), (u,))

    @classmethod
    def make_tuple(
        cls,
        quantities: tuple[bool | int | float | str | Enum, ...] | list[bool | int | float | str | Enum] | np.ndarray,
        no_unit: bool = False,
    ) -> tuple[tuple[bool | int | float | str | Enum, ...], tuple["Unit", ...]]:
        """Make a tuple of values and Unit objects from the tuple of quantities, using make()."""
        values: list[bool | int | float | str | Enum] = []
        units: list[Unit] = []
        for q in quantities:
            val, u = cls.make(q, no_unit=no_unit)
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
            u.to_base = bu.to_base  # link the functions
            u.from_base = bu.from_base  # link the functions
            u.du = None if bu.du is None else f"{bu.du}/{tu}"
            units.append(u)
        values = [0.0] * len(baseunits)
        return (tuple(values), tuple(units))

    def compatible(
        self, quantity: bool | int | float | str | Enum, no_unit: bool = False, strict: bool = True
    ) -> tuple[bool, bool | int | float | str | Enum]:
        """Check whether the supplied quantity 'q' is compatible with this unit.
        If strict==True, the supplied quantity shall be in display units.
        """
        _q, _unit = Unit.make(quantity, no_unit=no_unit)
        q = _q[0]
        unit = _unit[0]
        # no explicit unit needed when the quantity is 0 or inf (anything compatible)
        if (
            (q == 0 or q == float("inf") or q == float("-inf"))  # 0, +/-inf with any unit
            or (strict and self.u == unit.u and self.du == unit.du)
            or (not strict and self.u == unit.u)
        ):
            return (True, q)
        else:
            return (False, q)
