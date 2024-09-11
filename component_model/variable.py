from __future__ import annotations

import xml.etree.ElementTree as ET  # noqa: N817
from enum import Enum, IntFlag
from functools import partial
from math import acos, atan2, cos, degrees, radians, sin, sqrt
from typing import Any, Callable, TypeAlias

import numpy as np
from pint import Quantity  # management of units
from pythonfmu.enums import Fmi2Causality as Causality  # type: ignore
from pythonfmu.enums import Fmi2Initial as Initial  # type: ignore
from pythonfmu.enums import Fmi2Variability as Variability  # type: ignore
from pythonfmu.variables import ScalarVariable  # type: ignore

from .caus_var_ini import check_causality_variability_initial, use_start
from .logger import get_module_logger
from .utils import xml_to_python_val

logger = get_module_logger(__name__, level=0)
PyType: TypeAlias = str | int | float | bool | Enum
Numeric: TypeAlias = int | float
Compound: TypeAlias = tuple | list | np.ndarray


class Check(IntFlag):
    """Flags to denote how variables should be checked with respect to units and range.
    The aspects are indepent, but can be combined in the Enum through |.

    * none:     neither units nor ranges are expected or checked.
    * unitNone: only numbers without units expected when new values are provided.
      If units are provided during initialization, these should be base units (SE), i.e. unit and display are the same.
    * u_all:    expect always quantity and number and convert internally to base units (SE). Provide output as display
    * units:    flag to filter only on units, e.g ck & Check.units
    * r_none:   no range is provided or checked
    * r_check:  range is provided and checked
    * ranges:  flag to filter on range, e.g. ck & Check.ranges
    * all:     short for u_all | r_check
    """

    none = 0
    u_none = 0
    u_all = 1
    units = 1
    r_none = 0
    r_check = 2
    ranges = 2
    all = 3


def linear(x: float, b: float, a: float = 0.0):
    return a + b * x


# Some special error classes
class VariableInitError(Exception):
    """Special error indicating that something is wrong with the variable definition."""

    pass


class VariableRangeError(Exception):
    """Special Exception class signalling that a value is not within the range."""

    pass


class VariableUseError(Exception):
    """Special Exception class signalling that variable use was not in accordance with settings."""

    pass


class Variable(ScalarVariable):
    """Interface variable of an FMU. Can be a (python type) scalar variable. Extensions cover arrays (e.g. numpy array).
    The class extends pythonfmu.ScalarVariable, not using the detailed types (Real, Integer, ...), as these are handled internally.
    The recommended way to instantiate a Variable is through string values (with units for start and rng),
    but also dimensionless quantities and explicit class types are accepted.

    For proper understanding and usage the following should be noted:

    #. The Variable value is always owned by a model (see `self.model`).
    #. The current value of the variable is only directly accessible by the model through an attribute with the same name as `self.name`.
       Other access to the value is achieved through the `self.getter()` and the `self.setter( v)` + `self.on_set(v)` functions.
    #. It is recommended to register the Variable object as _`self.name` within the owner (model or sub-object of model),
       i.e. provide access as private object.
       In addition the model has access through the OrderedDict `.vars` ( {value_reference : variable object, ...})
    #. Compound variables (e.g. numpy arrays), should always allow setting of single elements.
       Due to limitations in fmi2 all variables are translated to ScalarVariables.
       The setter function therefore adds changed variables to a dirty dict and uses on_set on the fully changed array.
       The on_set() method can be used to perform post-setting activities and are always performed on the whole 'vector'.
    #. _display is set to None if it is equal to _unit or if dimensionless units are used,
        as a quick signal that no conversion is needed.

    Args:
        model (obj): The model object where this variable relates to. Use model.add_variable( name, ...) to define variables
        name (str): Variable name, unique for whole FMU !!and registered in the model for direct value access!!
        description (str) = None: Optional description of variable
        causality (str) = 'parameter': The causality setting as string
        variability (str) = 'fixed': The variability setting as string
        initial (str) = None: Definition how the variable is initialized. Provide this explicitly if the default value is not suitable.
        typ (type)=None: The type of variable to expect as start and value. Since initial values are often set with strings (with units, see below), this is set explicitly.
           If None, _typ is set to Enum/str if derived from these after disection or float if a number. 'int' is not automatically detected.
        start (PyType): The initial value of the variable.

           Optionally, the unit can be included, providing the initial value as string, evaluating to quantity of type typ a display unit and base unit.
           Note that the quantities are always converted to standard units of the same type, while the display unit may be different, i.e. the preferred user communication.
        rng (tuple) = (): Optional range of the variable in terms of a tuple of the same type as initial value. Can be specified with units (as string).

           * If an empty tuple is specified, the range is automatically determined.
             Note that it is thus not possible to automatically set single range elements (lower,upper) automatically
           * If None is specified, the initial value is chosen, i.e. no range. Applies to whole range tuple or to single elements (lower,upper)
           * For some variable types (e.g. str) no range is expected.

        annotations (dict) = None: Optional variable annotations provided as dict
        value_check (VarCheck) = VarCheck=VarCheck.r_check|VarCheck.u_all: Setting for checking of units and range according to VarCheck.
          The two aspects should be set with OR (|)
        fullInit (bool) = True: Optional possibility to stop the initialization of single variables, where this does not make sense for derived, compound variables
        on_step (callable) = None: Optonal possibility to register a function of (currentTime, dT) to be run during Model.do_step,
           e.g. if the variable represents a speed, the object can be translated speed*dT, if |speed|>0
        on_set (callable) = None: Optional possibility to specify a pre-processing function of (newVal) to be run when the variable is initialized or changed.
           This is useful for conditioning of input variables, so that calculations can be done once after a value is changed and do not need to be repeated on every simulation step.
           If given, the function shall apply to a value as expected by the variable (e.g. if there are components) and after unit conversion and range checking.
           The function is completely invisible by the user specifying inputs to the variable.

    .. todo:: Warnings on used default values which should be provided explicitly to conform to RP-0513
    .. limitation:: Limitation test
    .. assumption:: Assumption test
    .. requirement:: Requirement test
    """

    def __init__(
        self,
        model,
        name: str,
        description: str = "",
        causality: str | None = "parameter",
        variability: str | None = "fixed",
        initial: str | None = None,
        typ: type | None = None,
        start: PyType | Compound | None = None,
        rng: tuple | None = tuple(),
        annotations: dict | None = None,
        value_check: Check = Check.all,
        on_step: Callable | None = None,
        on_set: Callable | None = None,
        owner: Any | None = None,
    ):
        self.model = model
        self._causality, self._variability, self._initial = check_causality_variability_initial(
            causality, variability, initial
        )
        assert all(
            x is not None for x in (self._causality, self._variability, self._initial)
        ), f"Combination causality {self._causality}, variability {self._variability}, initial {self._initial} is not allowed"
        super().__init__(name=name, getter=self.getter, setter=self.setter)  # the other properties are done here!
        self.local_name: str
        if owner is None:
            self.owner = self.model
        else:
            self.owner = owner
            if hasattr(owner, "name") and self.local_name.startswith(owner.name + "_"):
                self.local_name = self.local_name[len(owner.name + "_") :]

        self._annotations = annotations
        self._check = value_check  # unique for all elements in compound variables
        self._typ = typ  # preliminary. Will be adapted if not explicitly provided (None)

        self.on_step = on_step  # hook to define a function of currentTime and time step dT,
        # to be performed during Model.do_step for input variables
        self.on_set = on_set
        # Note: the _len is a central property, distinguishing scalar and compound variables.

        self._start: tuple
        # First we check for str (since these are also iterable), then we can check for the presence of __getitem__
        # Determine the (element) type (unique for all elements in compound variables)
        if self._typ == str:  # explicit free string
            self._len = 1
            self.unit = "dimensionless"
            self.display = None
            self.range = ("", "")  # just a placeholder. Strings are not range checked
            self.start = ("",) if start is None else (str(start),)
        else:
            # if type is provided and no (initial) value. We set a default value of the correct type as 'example' value
            assert start is not None, "The start value is mandatory, at least for type and unit determination"
            _start, _unit, _display = self._disect_unit(start)  # do that first. units included as str!
            self.start = _start
            self.unit = _unit
            self.display = _display
            self._len = len(self._start)
            if self._typ is None:  # try to adapt using start
                self._typ = self.auto_type(self._start)
            if self._len > 1:  # make sure that all _start elements have the same type
                self._start = tuple(self._typ(self._start[i]) for i in range(self._len))
            self.range = self._init_range(rng)

        if not self.check_range(self._start):  # range checks of initial value
            raise VariableInitError(f"The provided value {self._start} is not in the valid range {self._range}")
        self.model.register_variable(self, self.start)  # register in model and return index
        # disable super() functions and properties which are not in use here
        self.to_xml = None

    # External access to read-only variables:
    def __len__(self) -> int:
        return self._len  # This works also compound variables, as long as _len is properly set

    @property
    def start(self):
        return self._start

    @start.setter
    def start(self, val):
        if isinstance(val, (str, int, float, bool, Enum)):
            self._start = (val,)
        elif isinstance(val, (tuple, list, np.ndarray)):
            self._start = tuple(val)
        else:
            raise VariableInitError(f"Unallowed start value setting {val} for variable {self.name}") from None

    @property
    def unit(self):
        return self._unit

    @unit.setter
    def unit(self, val):
        if isinstance(val, (tuple, list)):
            self._unit = tuple(val)
        elif isinstance(val, str):
            self._unit = (val,)
        else:
            raise VariableInitError(f"Unallowed unit setting {val} for variable {self.name}") from None

    @property
    def display(self):
        return self._display

    @display.setter
    def display(self, val):
        if val is None or (isinstance(val, tuple) and isinstance(val[0], str)):  # single variable
            self._display = (val,)
        elif isinstance(val, tuple) and (val[0] is None or isinstance(val[0], (tuple))):  # compound variable
            self._display = tuple(val)
        else:
            raise VariableInitError(f"Unallowed display setting {val} for variable {self.name}") from None

    @property
    def range(self):
        return self._range

    @range.setter
    def range(self, val):
        if isinstance(val, tuple) and isinstance(val[0], tuple):  # compound variable
            self._range = val
        elif isinstance(val, tuple) and all(isinstance(val[i], (int, float, bool, Enum, str)) for i in range(2)):
            self._range = (val,)

    @property
    def typ(self):
        return self._typ

    @property
    def check(self):
        return self._check

    @property
    def causality(self) -> Causality:
        return self._causality

    @property
    def variability(self) -> Variability:
        return self._variability

    @property
    def initial(self) -> Initial:
        return self._initial

    #     def __repr__(self):
    #         if self._len == 1:
    #             return f"Variable {self.name}. Initial: {self._start[0]}. Current {self.getter()} [{self.unit[0]}]."
    #         else:
    #             return f"Variable {self.name}. Initial: {self._start}. Current {self.getter()} [{self.unit}]."

    def setter(self, value: PyType | Compound, idx: int | None = None):
        """Set the value (input to model from outside), including range checking and unit conversion.

        For compound values, the whole 'array' must be provided,
        but elements which remain unchanged can be replaced by None.

        Alternatively, the index of compound variables can be provided explicitly.
        """
        assert self._typ is not None, "Need a proper type at this stage"
        if self._len == 1 and not isinstance(value, (tuple, list, np.ndarray)):
            value = [value]
            assert idx is None or idx == 0, f"Invalid idx {idx} for scalar"

        if issubclass(self._typ, Enum):  # Enum types are supplied as int. Convert
            for i in range(self._len):
                value[i] = self._typ(value[i])  # type: ignore

        if self._check & Check.units:  #'value' expected as displayUnit. Convert to unit
            if isinstance(idx, int):  # explicit index of single value
                if self._display[idx] is not None:
                    value = self.display[idx][1](value)
            else:
                if isinstance(value, tuple):  # tuples cannot be changed
                    value = list(value)
                for i in range(self._len):
                    if value[i] is not None and self._display[i] is not None:  # type: ignore
                        value[i] = self.display[i][1](value[i])  # type: ignore
        if self._check & Check.ranges:
            if not self.check_range(value, idx):
                raise VariableRangeError(f"set(): Value {value} outside range.") from None

        if self._len == 1:
            setattr(self.owner, self.local_name, value[0] if self.on_set is None else self.on_set(value[0]))  # type: ignore
        elif isinstance(idx, int):
            if value is not None:
                getattr(self.owner, self.local_name)[idx] = value
                if self.on_set is not None:
                    self.model.dirty_ensure(self)
        else:
            setattr(self.owner, self.local_name, value if self.on_set is None else self.on_set(value))

    def getter(self):
        """Get the value (output a value from the model), including range checking and unit conversion.
        The return value can be indexed/sliced to get elements of compound variables.
        """
        assert self._typ is not None, "Need a proper type at this stage"

        if self._len == 1:
            value = getattr(self.owner, self.local_name)
            if issubclass(self._typ, Enum):  # native Enums do not exist in FMI2. Convert to int
                value = value.value
            elif not isinstance(value, self._typ):  # other type conversion
                value = self._typ(value)
            if self._check & Check.units:  # Convert 'value' display.u -> base unit
                if self._display[0] is not None:
                    value = self.display[0][2](value)

        else:  # compound variable
            value = list(getattr(self.owner, self.local_name))  # make value available as copy
            if issubclass(self._typ, Enum):  # native Enums do not exist in FMI2. Convert to int
                for i in range(self._len):
                    value[i] = value[i].value
            else:
                for i in range(self._len):  # check whether conversion to _typ is necessary
                    if not isinstance(value[i], self._typ):
                        value[i] = self._typ(value[i])
            if self._check & Check.units:  # Convert 'value' display.u -> base unit
                for i in range(self._len):
                    if self._display[i] is not None:
                        value[i] = self.display[i][2](value[i])

        if self._check & Check.ranges and not self.check_range(value):
            raise VariableRangeError(f"getter(): Value {value} outside range.") from None
        return value

    def _init_range(self, rng: tuple | None) -> tuple:
        """Initialize the variable range(s) of the variable
        The _start and _unit shall exist when calling this.

        Args:
            rng (tuple): The tuple of range tuples.
              Always for the whole variable with scalar variables packed in a singleton

        """
        assert hasattr(self, "_start") and hasattr(self, "_unit"), "Missing self._start / self._unit"
        assert isinstance(self._typ, type), "init_range(): Need a defined _typ at this stage"
        # Configure input. Could be None, () or (min,max) of scalar
        if rng is None or rng == tuple() or (self._len == 1 and len(rng) == 2):
            rng = (rng,) * self._len

        _range = []
        for idx in range(self._len):  # go through all elements
            _rng = rng[idx]
            if _rng is None:  # => no range. Used for compound variables if not all elements have a range
                _range.append((self._start[idx], self._start[idx]))  # no range
            elif isinstance(_rng, tuple) and not len(_rng):  # empty tuple => try automatic range
                _range.append(self._auto_extreme(self._start[idx]))
            elif isinstance(_rng, tuple) and len(_rng) == 2:  # normal range as 2-tuple
                i_range = []  # collect range as list
                for r in _rng:
                    if r is None:  # no range => fixed to initial value
                        q = self._start[idx]
                    else:
                        q, u, du = self._disect_unit(r)
                        # no explicit unit needed when the quantity is 0 or inf
                        if (q == 0 or q == float("inf") or q == float("-inf")) and u == "dimensionless":
                            u = self._unit[idx]
                        elif self._unit[idx] != u:
                            raise VariableInitError(
                                f"The supplied range value {str(r)} does not conform to the unit type {self._unit[idx]}"
                            )
                        elif du is not None and self._display[idx] is not None and du[0] != self._display[idx][0]:
                            raise VariableInitError(f"Range unit {du[0]} != start {self._display[idx][0]}!")
                    if self._display[idx] is not None:  # Range in display units!
                        q = self._display[idx][2](q)
                    i_range.append(q)

                try:  # check variable type
                    i_range = [self._typ(x) for x in i_range]
                except Exception as err:
                    raise VariableRangeError(f"Incompatible types range {rng} - {self.start}") from err
                assert all(isinstance(x, self._typ) for x in i_range)
                _range.append(tuple(i_range))  # type: ignore
            else:
                raise AssertionError(f"init_range(): Unhandled range argument {rng}")
        return tuple(_range)

    def check_range(self, value: PyType | Compound | None, idx: int | None = None, disp: bool = True) -> bool:
        """Check the provided 'value' with respect to the range.

        Args:
            value (PyType|Compound): the value to check. Scalars may be wrapped as vector
            disp (bool) = True: denotes whether the value is expected in display units (default) or units

        Returns
        -------
            True/False with respect to whether val is the right type and is within range.
        """
        assert self._typ is not None, "Need a defined type at this stage"
        if self._len == 1 and idx is None:
            idx = 0
        if isinstance(value, str):  # no range checking on strings
            return self._typ == str
        elif self._len > 1 and idx is None:  # check all components
            assert isinstance(value, (tuple, list, np.ndarray)) and len(value) == self._len, f"{value} has no elements"
            return all(self.check_range(value[i], i) for i in range(self._len))
        else:  # single component check
            assert idx is not None, "Need a proper idx here"
            if isinstance(value, (tuple, list, np.ndarray)):
                if self._len == 1:
                    idx = 0
                value = value[idx]
            if value is None:  # denotes unchanged values (of compound variables)
                return True
            if self._typ != type(value):
                try:
                    value = self._typ(value)  # try to cast the value
                except Exception:  # give up
                    return False
            # special types (str checked above):
            if self._typ == str:  # no range checking on str
                return True
            elif self._typ == bool:
                return isinstance(value, bool)
            elif isinstance(value, Enum):
                return isinstance(value, self._typ)

            elif isinstance(value, (int, float)) and all(isinstance(x, (int, float)) for x in self._range[idx]):
                if not disp and self._display[idx] is not None:  # check an internal unit value
                    value = self._display[idx][2](value)
                return self._range[idx] is None or self._range[idx][0] <= value <= self._range[idx][1]  # type: ignore
            else:
                raise VariableUseError(f"check_range(): value={value}, type={self.typ}, range={self.range}") from None

    def fmi_type_str(self, val: PyType) -> str:
        """Translate the provided type to a proper fmi type and return as string.
        See types defined in schema fmi2Unit.xsd.
        """
        if self._typ == bool:
            return "true" if val else "false"
        else:
            return str(val)

    @classmethod
    def auto_type(cls, val: PyType | Compound, allow_int: bool = False):
        """Determine the Variable type from a provided example value.
        Since variables can be initialized using strings with units,
        the type can only be determined when the value is disected.
        Moreover, the value may indicate an integer, while the variable is designed a float.
        Therefore int Variables must be explicitly specified.
        """
        assert val is not None, "'val is None'!"
        if isinstance(val, (tuple, list, np.ndarray)):
            types = [cls.auto_type(x, allow_int) for x in val]
            typ = None
            for t in types:
                if t is not None and typ is None:
                    typ = t
                elif t is not None and typ is not None:
                    if t == typ:
                        pass
                    elif t != typ:  # identify the super-type
                        if issubclass(t, typ):  # is a sub-class. Ok
                            pass
                        elif issubclass(typ, t):
                            typ = t
                        elif typ == float and t == int:  # we allow that, even if no subclass
                            pass
                        elif typ == int and t == float:  # we allow that, even if no subclass
                            typ = float
                    else:
                        raise VariableInitError(f"Incompatible variable types {typ}, {t} in {val}") from None
                else:
                    raise ValueError(f"auto_type(). Unhandled {t}, {typ}")
            return typ
        else:  # single value
            if isinstance(val, bool):
                return bool
            elif allow_int:
                return type(val)
            elif not allow_int and isinstance(val, (int, float)):
                return float
            else:
                return type(val)

    @classmethod
    def _auto_extreme(cls, var: PyType) -> tuple:
        """Return the extreme value of the variable.

        Args:
            var: the variable for which to determine the extremes. Represented by an instantiated object
        Returns:
            A tuple containing the minimum and maximum value the given variable can have
        """
        if isinstance(var, bool):
            return (False, True)
        elif isinstance(var, float):
            return (float("-inf"), float("inf"))
        elif isinstance(var, int):
            raise VariableInitError(f"Range must be specified for int variable {cls} or use float.")
        elif isinstance(var, Enum):
            return (min(x.value for x in type(var)), max(x.value for x in type(var)))
        else:
            return tuple()  # return an empty tuple (no range specified, e.g. for str)

    def _disect_unit(self, quantity: PyType | Compound) -> tuple:
        """Disect the provided quantity in terms of magnitude and unit, if provided as string.
        If another type is provided, dimensionless units are assumed.

        Args:
            quantity (PyType): the quantity to disect. Should be provided as string, but also the trivial cases (int,float,Enum) are allowed.
            A free string should not be used and leads to a warning
        Returns:
            the magnitude in base units, the base unit and the unit as given together with the conversion factor (from display to baseUnit)
        """
        if isinstance(quantity, (tuple, list, np.ndarray)):  # handle composit values
            _val, _ub, _disp = [], [], []
            for q in quantity:  # disect components and collect results
                v, u, d = self._disect_unit(q)
                _val.append(v)
                _ub.append(u)
                _disp.append(d)
            return (tuple(_val), tuple(_ub), None if _disp is None else tuple(_disp))

        elif isinstance(quantity, str):  # only string variable make sense to disect
            assert self.model.ureg is not None, f"UnitRegistry not found, while providing units: {quantity}"
            try:
                q = self.model.ureg(quantity)  # parse the quantity-unit and return a Pint Quantity object
                if isinstance(q, (int, float)):
                    return q, "", None  # integer or float variable with no units provided
                elif isinstance(q, Quantity):  # pint.Quantity object
                    # transform to base units ('SI' units). All internal calculations will be performed with these
                    val, ub, display = self._get_transformation(q)
                else:
                    raise VariableInitError(f"Unknown quantity {quantity} to disect") from None
            # no recognized units. Assume a free string. ??Maybe we should be more selective about the exact error type:
            except Exception as warn:
                logger.warning(f"Unhandled quantity {quantity}: {warn}. A str? Set explicit 'typ=str'.")
                val, ub, display = (str(quantity), "", None)  # type: ignore
        else:
            val, ub, display = (quantity, "dimensionless", None)  # type: ignore
        if self._typ is not None and type(val) != self._typ:  # check variable type
            try:  # try to convert the magnitude to the correct type.
                val = self._typ(val)
            except Exception as err:
                raise VariableInitError(f"Value {val} is not of the correct type {self._typ}") from err
        return val, ub, display

    def _get_transformation(self, q: Quantity) -> tuple[float, str, tuple | None]:
        """Identity base units and calculate the transformations between display and base units."""
        qb = q.to_base_units()
        val = qb.magnitude  # Note: numeric types are not converted, e.g. int to float
        if qb.units == q.units:  # no conversion
            return (val, str(qb.units), None)
        else:  # calculate the conversion functions
            # we generate a second value and calculate the straight line conversion function
            # did not find a better way in pint
            q2 = self.model.ureg.Quantity(10.0 * (q.magnitude + 10.0), q.units)
            qb2 = q2.to_base_units()
            a = (qb.magnitude * q2.magnitude - qb2.magnitude * q.magnitude) / (q2.magnitude - q.magnitude)
            b = (qb2.magnitude - qb.magnitude) / (q2.magnitude - q.magnitude)
            if abs(a) < 1e-9:  # multiplicative conversion
                if abs(b - 1.0) < 1e-9:  # unit and display unit are compatible. No transformation
                    return (val, str(qb.units), None)
                to_base = partial(linear, b=b)
                from_base = partial(linear, b=1.0 / b)
            else:  # there is a constant (e.g. Celsius to Fahrenheit)
                to_base = partial(linear, b, a)
                from_base = partial(linear, b=1.0 / b, a=-a / b)
            return (val, str(qb.units), (str(q.units), to_base, from_base))

    def xml_scalarvariables(self):
        """Generate <ScalarVariable> XML code with respect to this variable and return xml element.
        For compound variables, all elements are included.
        Note that ScalarVariable attributes should all be listed in __attrs dictionary.
        Since we do not use the derived classes Real, ... we need to generate the detailed variable definitions here.
        The following attributes are so far not supported: declaredType, derivative, reinit.

        Used properties:
            * typ: the type can be explicitly provided (for derived variables), otherwise self._typ is used
            * start: a start can be explicitly provided (for components of derived variables)
            * valueReference: For compound variables this is provided explicitly for >0 elements.
            * range: a range tuple can be explicitly provided
            * unit: a unit can be explicitly provided
            * display: a displayUnit can be explicitly provided

        Returns
        -------
            List fo ScalarVariable xml elements
        """

        def substr(alt1: str, alti: str):
            return alt1 if self._len == 1 else alti

        declaredType = {"int": "Integer", "bool": "Boolean", "float": "Real", "str": "String", "Enum": "Enumeration"}[
            self.typ.__qualname__
        ]  # translation of python to FMI primitives. Same for all components
        do_use_start = use_start(self._causality, self._variability, self._initial)
        svars = []
        for i in range(self._len):
            sv = ET.Element(
                "ScalarVariable",
                {
                    "name": self.name + substr("", f"[{i}]"),
                    "valueReference": str(self.value_reference + i),
                    "causality": self.causality.name,
                    "variability": self.variability.name,
                },
            )
            if self._initial != Initial.none:  # none is not part of the FMI2 specification
                sv.attrib.update({"initial": self.initial.name})
            if self.description is not None:
                sv.attrib.update({"description": self.description + substr("", f", [{i}]")})
            if self._annotations is not None and i == 0:
                sv.append(ET.Element("annotations", self._annotations))
            #             if self.display is None or (self._len>1 and self.display[i] is None):
            #                 "display" = (self.unit, 1.0)

            # detailed variable definition
            varInfo = ET.Element(declaredType)
            if do_use_start:  # a start value is to be used
                varInfo.attrib.update({"start": self.fmi_type_str(self.start[i])})
            if declaredType in ("Real", "Integer", "Enumeration"):  # range to be specified
                xMin = self.range[i][0]
                if declaredType != "Real" or xMin > float("-inf"):
                    varInfo.attrib.update({"min": str(xMin)})
                else:
                    varInfo.attrib.update({"unbounded": "true"})
                xMax = self.range[i][1]
                if declaredType != "Real" or xMax < float("inf"):
                    varInfo.attrib.update({"max": str(xMax)})
                else:
                    varInfo.attrib.update({"unbounded": "true"})
            if declaredType == "Real":  # other attributes apply only to Real variables
                varInfo.attrib.update({"unit": self.unit[i]})
                if self.display is not None and self.display[i] is not None and self.unit[i] != self.display[i][0]:
                    varInfo.attrib.update({"displayUnit": self.display[i][0]})
            sv.append(varInfo)
            svars.append(sv)
        return svars


# Utility functions for handling special variable types
def spherical_to_cartesian(vec: np.ndarray | tuple, asDeg: bool = False) -> np.ndarray:
    """Turn spherical vector 'vec' (defined according to ISO 80000-2 (r,polar,azimuth)) into cartesian coordinates."""
    if asDeg:
        theta = radians(vec[1])
        phi = radians(vec[2])
    else:
        theta = vec[1]
        phi = vec[2]
    sinTheta = sin(theta)
    cosTheta = cos(theta)
    sinPhi = sin(phi)
    cosPhi = cos(phi)
    r = vec[0]
    return np.array((r * sinTheta * cosPhi, r * sinTheta * sinPhi, r * cosTheta))


def cartesian_to_spherical(vec: np.ndarray | tuple, asDeg: bool = False) -> np.ndarray:
    """Turn the vector 'vec' given in cartesian coordinates into spherical coordinates.
    (defined according to ISO 80000-2, (r, polar, azimuth)).
    """
    r = np.linalg.norm(vec)
    if vec[0] == vec[1] == 0:
        if vec[2] == 0:
            return np.array((0, 0, 0), dtype="float")
        else:
            return np.array((r, 0, 0), dtype="float")
    elif asDeg:
        return np.array((r, degrees(acos(vec[2] / r)), degrees(atan2(vec[1], vec[0]))), dtype="float64")
    else:
        return np.array((r, acos(vec[2] / r), atan2(vec[1], vec[0])), dtype="float")


def cartesian_to_cylindrical(vec: np.ndarray | tuple, asDeg: bool = False) -> np.ndarray:
    """Turn the vector 'vec' given in cartesian coordinates into cylindrical coordinates.
    (defined according to ISO, (r, phi, z), with phi right-handed wrt. x-axis).
    """
    phi = atan2(vec[1], vec[0])
    if asDeg:
        phi = degrees(phi)
    return np.array((sqrt(vec[0] * vec[0] + vec[1] * vec[1]), phi, vec[2]), dtype="float")


def cylindrical_to_cartesian(vec: np.ndarray | tuple, asDeg: bool = False) -> np.ndarray:
    """Turn cylinder coordinate vector 'vec' (defined according to ISO (r,phi,z)) into cartesian coordinates.
    The angle phi is measured with respect to x-axis, right hand.
    """
    phi = radians(vec[1]) if asDeg else vec[1]
    return np.array((vec[0] * cos(phi), vec[0] * sin(phi), vec[2]), dtype="float")


def quantity_direction(quantityDirection: tuple, asSpherical: bool = False, asDeg: bool = False) -> np.ndarray:
    """Turn a 4-tuple, consisting of quantity (float) and a direction 3-vector to a direction 3-vector,
    where the norm denotes the direction and the length denotes the quantity.
    The return vector is always a cartesian vector.

    Args:
        quantityDirection (tuple): a 4-tuple consisting of the desired length of the resulting vector (in standard units (m or m/s))
           and the direction 3-vector (in standard units)
        asSpherical (bool)=False: Optional possibility to provide the input direction vector in spherical coordinates
        asDeg (bool)=False: Optional possibility to provide the input angle (of spherical coordinates) in degrees. Only relevant if asSpherical=True
    """
    if quantityDirection[0] < 1e-15:
        return np.array((0, 0, 0), dtype="float")
    if asSpherical:
        direction = spherical_to_cartesian(quantityDirection[1:], asDeg)  # turn to cartesian coordinates, if required
    else:
        direction = np.array(quantityDirection[1:], dtype="float")
    n = np.linalg.norm(direction)  # normalize
    return quantityDirection[0] / n * direction


def variables_from_fmu(model, el: ET.Element | None, sep: str = "["):
    """From the supplied model object and the <ModelVariables> el subtree identify and define all variables.
    .. toDo:: implement unit and displayUnit handling + <UnitDefinitions>.
    """

    def range_from_fmu(el: ET.Element):
        """From the variable type sub-element (e.g. <Real>) of <ScalarVariable> deduce the variable range of a ScalarVariable."""
        if el.attrib.get("unbounded", "true"):
            return tuple()
        elif "min" in el.attrib and "max" in el.attrib:
            return (el.attrib["min"], el.attrib["max"])
        elif "min" in el.attrib and el.tag == "Real":
            return (el.attrib["min"], float("inf"))
        elif "max" in el.attrib and el.tag == "Real":
            return (float("-inf"), el.attrib["max"])
        else:
            raise AssertionError(
                f"Invalid combination of attributes with respect to variable range. Type:{el.tag}, attributes: {el.attrib}"
            )

    def rsplit_sep(txt: str, sep: str = sep):
        if sep in txt:
            base, sub = txt.rsplit(sep, maxsplit=1)
            if sep == "[":
                sub = sub.rsplit("]", maxsplit=1)[0]
            elif sep == "(":
                sub = sub.rsplit(")", maxsplit=1)[0]
            elif sep == "{":
                sub = sub.rsplit("}", maxsplit=1)[0]
            return (base, sub)
        else:
            return (txt, "")

    idx = 0
    while True:
        if el is None:
            break
        var = el[idx]
        base, sub = rsplit_sep(var.attrib["name"])
        length = 1
        _causality, _variability, _initial = (
            var.get("causality", "local"),
            var.get("variability", "continuous"),
            var.get("initial", None),
        )
        _typ = xml_to_python_val(var[0].tag)
        if len(base) and len(sub) and sub.isnumeric() and int(sub) == 0:  # assume first element of a compound variable
            for i in range(idx + 1, len(el)):  # collect the other elements of this compound variable
                v = el[i]
                b, s = rsplit_sep(v.attrib["name"])
                if not (
                    len(b)
                    and len(s)
                    and s.isnumeric()
                    and b == base
                    and int(s) == i - idx
                    and v.attrib["causality"] == _causality
                    and v.attrib["variability"] == _variability
                    and v.get("initial", None) == _initial
                    and v[0].tag == var[0].tag
                ):  # this element does not fit in the compound variable
                    length = i - idx
                    break
        #        if length == 1:  # a scalar
        _var = Variable(
            model,
            base,
            description=var.attrib["description"],
            causality=_causality,
            variability=_variability,
            initial=_initial,
            typ=_typ,
            start=var[0].attrib.get("start", None),
            rng=range_from_fmu(var[0]),
        )

        #         else:  # an array
        #             _var = VariableNP(
        #                 model,
        #                 base,
        #                 description=var.attrib["description"],
        #                 causality=_causality,
        #                 variability=_variability,
        #                 initial=_initial,
        #                 typ=_typ,
        #                 start=tuple(v[0].attrib.get("start", None) for v in el[idx : idx + length]),
        #                 rng=tuple(range_from_fmu(v[0]) for v in el[idx : idx + length]),
        #             )
        idx += length
        if idx >= len(el):
            break
