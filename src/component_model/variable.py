from __future__ import annotations

import logging
import xml.etree.ElementTree as ET  # noqa: N817
from enum import Enum, IntFlag
from functools import partial
from math import acos, atan2, cos, degrees, radians, sin, sqrt
from typing import Any, Callable, Sequence, TypeAlias

import numpy as np
from pint import Quantity  # management of units
from pythonfmu.enums import Fmi2Causality as Causality  # type: ignore
from pythonfmu.enums import Fmi2Initial as Initial  # type: ignore
from pythonfmu.enums import Fmi2Variability as Variability  # type: ignore
from pythonfmu.variables import ScalarVariable  # type: ignore

from component_model.enums import check_causality_variability_initial, use_start
from component_model.variable_naming import ParsedVariable

logger = logging.getLogger(__name__)
PyType: TypeAlias = str | int | float | bool | Enum
Numeric: TypeAlias = int | float
Compound: TypeAlias = tuple[PyType, ...] | list[PyType] | np.ndarray


class Check(IntFlag):
    """Flags to denote how variables should be checked with respect to units and range.
    The aspects are indepent, but can be combined in the Enum through | or &.

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
    """Interface variable of an FMU. Can be a (python type) scalar variable or a numpy array.
    The class extends pythonfmu.ScalarVariable, not using the detailed types (Real, Integer, ...),
    as the type is handled internally (automatically or explicit, see below).
    The recommended way to instantiate a Variable is through string values (with units for start and rng),
    but also dimensionless quantities and explicit class types are accepted.

    For proper understanding and usage the following should be noted:

    #. The Variable value is per default owned by the related model (see `self.model`).
       Through the `owner` parameter this can be changes.
       In structured models like the crane_fmu this might be adequate.
    #. The current value of the variable directly accessible through the owner.
       Direct value access assumes always internal units (per default: SI units) and range checking is not performed.
    #. Other access to the value is achieved through the `self.getter()` and the `self.setter( v)` functions,
       which are also used by the FMU getxxx and setxxx functions (external access).
       Unit transformation and range checking is performed within getter() and setter(),
       i.e. the setter function assumes a value in display units and transforms it into internal units and
       the getter function assumes internal units and makes the value available as display units.
       For example an angle variable might be defined as `degrees` (display units),
       but will internally always be handled as `radians`.
    #. It is recommended to register the Variable object as _`self.name` within the owner (model or sub-object of model),
       i.e. provide access as private object. In addition the model has access to the variable object
       through the OrderedDict `.vars = {value_reference : variable object, ...}`.
    #. Compound variables (e.g. numpy arrays), should always allow setting of single elements.
       Due to limitations in fmi2 all variables are translated to ScalarVariables, using the syntax `np-variable[idx]`.
    #. For variables it is sometimes required to perform actions additional to the standard setter actions
       (unit trnsformation and range checking). Such actions can be defined through the `self.on_set()` hook.
       For compound variables `on_set` should only be run when all elements have received their new value.
       This is ensured through the internal `dirty` mechanism.
       `on_set()` should therefore always address the whole variable, not single elements.
    #. For variables it is sometimes convenient to perform a fixed action at each step of the simulation.
       This can be conveniently done through the on_step(time,dt) hook.

    Args:
        model (obj): The model object where this variable relates to. Use model.add_variable( name, ...) to define variables
        name (str): Variable name, unique for whole FMU.
        description (str) = None: Optional description of variable
        causality (str) = 'parameter': The causality setting as string
        variability (str) = 'fixed': The variability setting as string
        initial (str) = None: Optional definition how the variable is initialized. Provide this explicitly if the default value is not suitable.
        typ (type)=None: Optional explicit type of variable to expect as start and value.
           Since initial values are often set with strings (with units, see below), this is set explicitly.
           If None, _typ is set to Enum/str if derived from these after disection or float if a number. 'int' is not automatically detected.
        start (PyType): The initial value of the variable.

           Optionally, the unit can be included, providing the initial value as string, evaluating to quantity of type typ a display unit and base unit.
           Note that the quantities are always converted to standard units of the same type, while the display unit may be different, i.e. the preferred user communication.
        rng (tuple) = (): Optional range of the variable in terms of a tuple of the same type as initial value. Should be specified with units (as string).

           * If an empty tuple is specified, the range is automatically determined.
             That is only possible float or enum type variables, where the former evaluates to (-inf, inf).
             Maximum or minimum int values do not exist in Python, such that these always must be provided explicitly.
             It is not possible to set only one of the elements of the tuple automatically.
           * If None is specified, the initial value is chosen, i.e. no range.
             `None` can be applied to the whole tuple or to single elements of the tuple.
             E.g. (1,None) sets the range to (1, start)
           * For some variable types (e.g. str) no range is expected.

        annotations (dict) = None: Optional variable annotations provided as dict
        value_check (Check) = Check=Check.r_check|Check.u_all:
          Setting for checking of units and range according to Check.
          The two aspects should be set with OR (|),
          e.g. `Check.units | Check.r_none` leads to only units transformations but no range checking.
        on_step (callable) = None: Optional possibility to register a function of `(time, dt)` to be run during `.do_step`,
           e.g. if the variable represents a speed, the object can be translated `speed*dt, if |speed|>0`
        on_set (callable) = None: Optional possibility to specify a pre-processing function of (newval)
           to be run when the variable is initialized or changed.
           This is useful for conditioning of input variables, so that calculations can be done once after a value is changed
           and do not need to be repeated on every simulation step.
           If given, the function shall apply to the whole (vecor) variable,
           and after unit conversion and range checking.
           The function is completely invisible by the user specifying inputs to the variable.
        owner = None: Optional possibility to overwrite the default owner.
           If the related model uses structured variable naming this should not be necessary,
           but for flat variable naming within complex models (not recommended) ownership setting might be necessary.
        local_name (str) = None: Optional possibility to overwrite the automatic determination of local_name,
           which is used to access the variable value and must be a property of owner.
           This is convenient for example to link a derivative name to a variable if the default name der_<base-var>
           is not acceptable.
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
        local_name: str | None = None,
    ):
        self.model = model
        self._causality, self._variability, self._initial = check_causality_variability_initial(
            causality, variability, initial
        )
        assert all(x is not None for x in (self._causality, self._variability)), (
            f"Combination causality {self._causality}, variability {self._variability}, initial {self._initial} is not allowed"
        )
        super().__init__(name=name, description=description, getter=self.getter, setter=self.setter)

        parsed = ParsedVariable(name, self.model.variable_naming)
        if owner is None:
            oh = self.model.owner_hierarchy(parsed.parent)
            if owner is None:
                self.owner = oh[-1]
        else:
            self.owner = owner
        if local_name is None:
            if parsed.der > 0:  # is a derivative of 'var'
                self.local_name = f"der{parsed.der}_{parsed.var}"
                if not hasattr(self.owner, self.local_name):  # the derivative is not explicitly defined in the model
                    if parsed.der == 1:  # first derivative
                        self.local_name = f"der{parsed.der}_{parsed.var}"
                        setattr(self.owner, self.local_name, 0.0)
                        if on_step is None:
                            on_step = self.der1
                    else:
                        raise NotImplementedError("None-explicit higher order derivatives not implemented") from None
            else:
                self.local_name = parsed.var
        else:
            self.local_name = local_name

        self._annotations = annotations
        self._check = value_check  # unique for all elements in compound variables
        self._typ: type | None = typ  # preliminary. Will be adapted if not explicitly provided (None)

        self.on_step = on_step  # hook to define a function of currentTime and time step dT,
        # to be performed during Model.do_step for input variables
        self.on_set = on_set
        # Note: the _len is a central property, distinguishing scalar and compound variables.

        self._start: tuple
        # First we check for str (since these are also iterable), then we can check for the presence of __getitem__
        # Determine the (element) type (unique for all elements in compound variables)
        if self._typ is str:  # explicit free string
            self._len = 1
            self.unit = "dimensionless"
            self.display = None
            self.range = ("", "")  # just a placeholder. Strings are not range checked
            self.start = ("",) if start is None else (str(start),)
        else:
            # if type is provided and no (initial) value. We set a default value of the correct type as 'example' value
            assert start is not None, f"{self.name}: start value is mandatory, at least for type and unit determination"
            _start, _unit, _display = self._disect_unit(start)  # do that first. units included as str!
            self.start = _start
            self.unit = _unit
            self.display = _display
            self._len = len(self._start)
            if self._typ is None:  # try to adapt using start
                self._typ = self.auto_type(self._start)
            assert isinstance(self._typ, type)
            if self._len > 1:  # make sure that all _start elements have the same type
                self._start = tuple(self._typ(self._start[i]) for i in range(self._len))
            self.range = self._init_range(rng)

        if not self.check_range(self._start, disp=False):  # range checks of initial value
            raise VariableInitError(f"The provided value {self._start} is not in the valid range {self._range}")
        self.model.register_variable(self)
        try:
            setattr(self.owner, self.local_name, np.array(self.start, self.typ) if self._len > 1 else self.start[0])
        except AttributeError as _:  # can happen if a @property is defined for local_name, but no @local_name.setter
            pass

    def der1(self, current_time: float, step_size: float):
        der = self.getter()  # the current slope value
        if any(x != 0.0 for x in der):  # the value is (currently) set to > or < 0
            varname = self.local_name[5:]
            if len(der) == 1:
                assert isinstance(der[0], float)
                setattr(self.owner, varname, getattr(self.owner, varname) + der[0] * step_size)
            else:
                setattr(
                    self.owner,
                    varname,
                    np.array(getattr(self.owner, varname), float) + np.array(der, float) * step_size,
                )

    # disable super() functions and properties which are not in use here
    def to_xml(self) -> ET.Element:
        raise NotImplementedError("The function to_xml() shall not be used from component-model") from None

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
        return self._causality  # type: ignore

    @property
    def variability(self) -> Variability:
        return self._variability  # type: ignore

    @property
    def initial(self) -> Initial | None:
        return self._initial

    def setter(self, values: Sequence[int | float | bool | str | Enum] | np.ndarray, idx: int = -1):
        """Set the values (input to model from outside), including range checking and unit conversion.

        For compound values, the whole 'array' should be provided,
        but elements which remain unchanged can be replaced by None.
        Alternatively, single elements can be set by providing the index explicitly.
        """
        dvals: list[int | float | bool | str | Enum | None]
        logger.debug(f"SETTER0 {self.name}, {values}[{idx}] => {getattr(self.owner, self.local_name)}")
        is_ndarray = isinstance(values, np.ndarray)
        assert self._typ is not None, "Need a proper type at this stage"
        assert isinstance(values, (Sequence, np.ndarray)), "A sequence is expected as values"
        if idx == -1 and self._len == 0:  # the whole scalar
            idx = 0

        if issubclass(self._typ, Enum):  # Enum types are supplied as int. Convert
            for i in range(self._len):
                values[i] = self._typ(values[i])  # type: ignore

        if self._check & Check.ranges:  # do that before unit conversion, since range is stored in display units!
            if not self.check_range(values, idx):
                raise VariableRangeError(f"set(): values {values} outside range.") from None

        if self._check & Check.units:  #'values' expected as displayUnit. Convert to unit
            if idx >= 0:  # explicit index of single values
                if self._display[idx] is not None:
                    dvals = [self.display[idx][1](values[0])]
                else:
                    dvals = list(values)
            else:  # the whole array
                dvals = []
                for i in range(self._len):
                    if values[i] is None:  # keep the value
                        dvals.append(getattr(self.owner, self.local_name)[i])
                    elif self._display[i] is None:
                        dvals.append(values[i])
                    else:
                        dvals.append(self.display[i][1](values[i]))
        else:  # no unit issues
            if self._len == 1:
                dvals = [values[0] if values[0] is not None else getattr(self.owner, self.local_name)]
            else:
                dvals = [
                    values[i] if values[i] is not None else getattr(self.owner, self.local_name)[i]
                    for i in range(self._len)
                ]

        if self._len == 1:
            setattr(self.owner, self.local_name, dvals[0] if self.on_set is None else self.on_set(dvals[0]))  # type: ignore
        elif idx >= 0:
            if dvals[0] is not None:
                getattr(self.owner, self.local_name)[idx] = dvals[0]
                if self.on_set is not None:
                    self.model.dirty_ensure(self)
        else:  # the whole array
            if is_ndarray:  # Note: on_set might contain array operations
                arr: np.ndarray = np.array(dvals, self._typ)
                setattr(self.owner, self.local_name, arr if self.on_set is None else self.on_set(arr))
            else:
                setattr(self.owner, self.local_name, dvals if self.on_set is None else self.on_set(dvals))
        if self.on_set is None:
            logger.debug(f"SETTER {self.name}, {values}[{idx}] => {getattr(self.owner, self.local_name)}")

    def getter(self) -> Sequence[PyType]:
        """Get the value (output a value from the model), including range checking and unit conversion.
        The whole variable value is returned.
        The return a list of values. Can later be indexed/sliced to get elements of compound variables.
        """
        assert self._typ is not None, "Need a proper type at this stage"
        if self._len == 1:
            value = getattr(self.owner, self.local_name)
            if issubclass(self._typ, Enum):  # native Enums do not exist in FMI2. Convert to int
                value = value.value
            elif not isinstance(value, self._typ):  # other type conversion
                value = self._typ(value)  # type: ignore[call-arg]
            if self._check & Check.units:  # Convert 'value' display.u -> base unit
                if self._display[0] is not None:
                    value = self.display[0][2](value)
            values = [value]

        else:  # compound variable
            values = list(getattr(self.owner, self.local_name))  # make value available as copy
            if issubclass(self._typ, Enum):  # native Enums do not exist in FMI2. Convert to int
                for i in range(self._len):
                    values[i] = values[i].value
            else:
                for i in range(self._len):  # check whether conversion to _typ is necessary
                    if not isinstance(values[i], self._typ):
                        values[i] = self._typ(values[i])  # type: ignore[call-arg]
            if self._check & Check.units:  # Convert 'value' display.u -> base unit
                for i in range(self._len):
                    if self._display[i] is not None:
                        values[i] = self.display[i][2](values[i])

        if self._check & Check.ranges and not self.check_range(values, -1):
            raise VariableRangeError(f"getter(): Value {values} outside range.") from None
        return values

    def _init_range(self, rng: tuple | None) -> tuple:
        """Initialize the variable range(s) of the variable
        The _start and _unit shall exist when calling this.

        Args:
            rng (tuple): The tuple of range tuples.
              Always for the whole variable with scalar variables packed in a singleton
        """

        def ensure_display_limits(val: PyType, idx: int, right: bool):
            """Ensure that value is provided as display unit and that limits are included in range."""
            if self._display[idx] is not None:  # Range in display units!
                _val = self._display[idx]
                assert isinstance(_val, tuple)
                val = _val[2](val)
            if isinstance(val, float) and abs(val) != float("inf") and int(val) != val:
                if right:
                    val += 1e-15
                else:
                    val -= 1e-15
            return val

        assert hasattr(self, "_start") and hasattr(self, "_unit"), "Missing self._start / self._unit"
        assert isinstance(self._typ, type), "init_range(): Need a defined _typ at this stage"
        # Configure input. Could be None, () or (min,max) of scalar
        if rng is None or rng == tuple() or (self._len == 1 and len(rng) == 2):
            rng = (rng,) * self._len

        _range = []
        for idx in range(self._len):  # go through all elements
            _rng = rng[idx]
            if _rng is None:  # => no range. Used for compound variables if not all elements have a range
                _range.append(
                    (
                        ensure_display_limits(self._start[idx], idx, right=False),
                        ensure_display_limits(self._start[idx], idx, right=True),
                    )
                )  # no range
            elif isinstance(_rng, tuple) and not len(_rng):  # empty tuple => try automatic range
                _range.append(self._auto_extreme(self._start[idx]))
            elif isinstance(_rng, tuple) and len(_rng) == 2:  # normal range as 2-tuple
                i_range: list = []  # collect range as list
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
                        elif du is not None and self._display[idx] is not None and du[0] != self._display[idx][0]:  # type: ignore[index]
                            raise VariableInitError(f"Range unit {du[0]} != start {self._display[idx][0]}!")  # type: ignore[index]
                    q = ensure_display_limits(q, idx, len(i_range) > 0)
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

    def check_range_single(self, value: PyType | None, idx: int = 0, disp: bool = True) -> bool:
        """Check a single value."""
        assert idx >= 0, f"Need a proper idx here. Found {idx}"
        assert self._typ is not None
        if value is None:  # denotes unchanged values (of compound variables)
            return True
        if self._typ is not type(value):
            try:
                value = self._typ(value)  # try to cast the values
            except Exception:  # give up
                return False
        # special types
        if self._typ is str:  # no range checking on str
            return True
        elif self._typ is bool:
            return isinstance(value, bool)
        elif isinstance(value, Enum):
            assert self._typ is not None
            return isinstance(value, self._typ)

        elif isinstance(value, (int, float)) and all(isinstance(x, (int, float)) for x in self._range[idx]):
            if not disp and self._display[idx] is not None:  # check an internal unit values
                _val = self._display[idx]
                assert isinstance(_val, tuple)
                value = _val[2](value)
            return self._range[idx] is None or self._range[idx][0] <= value <= self._range[idx][1]  # type: ignore
        else:
            raise VariableUseError(f"check_range(): value={value}, type={self.typ}, range={self.range}") from None

    def check_range(self, values: Sequence[PyType | None] | np.ndarray, idx: int = 0, disp: bool = True) -> bool:
        """Check the provided 'values' with respect to the range.

        Args:
            values (Sequence[PyType]): the value(s) to check. Scalars are wrapped into a Sequence
            idx (int)=0: optional index of variable to check. -1 means all indices
            disp (bool) = True: denotes whether the values is expected in display units (default) or units

        Returns
        -------
            True/False with respect to whether values is the right type and is within range.
        """
        assert self._typ is not None, "Need a defined type at this stage"
        if self._len == 1 and idx == -1:
            idx = 0
        if isinstance(values[0], str):  # no range checking on strings
            return self._typ is str
        elif self._len > 1 and idx < 0:  # check all components
            assert isinstance(values, (Sequence, np.ndarray)) and len(values) == self._len, (
                f"Values {values} not sufficient. Need {self._len}"
            )
            return all(self.check_range_single(values[i], i, disp) for i in range(self._len))
        else:
            return self.check_range_single(values[0], idx, disp)

    def fmi_type_str(self, val: PyType) -> str:
        """Translate the provided type to a proper fmi type and return it as string.
        See types defined in schema fmi2Unit.xsd.
        """
        if self._typ is bool:
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
                        elif typ is float and t is int:  # we allow that, even if no subclass
                            pass
                        elif typ is int and t is float:  # we allow that, even if no subclass
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
        """Return the extreme values of the variable.

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
            the magnitude in base units, the base unit and the unit as given (display units),
            together with the conversion functions between the units.
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
        if self._typ is not None and type(val) is not self._typ:  # check variable type
            try:  # try to convert the magnitude to the correct type.
                val = self._typ(val)
            except Exception as err:
                raise VariableInitError(f"Value {val} is not of the correct type {self._typ}") from err
        return (val, ub, display)

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

        Returns
        -------
            List of ScalarVariable xml elements
        """

        def substr(alt1: str, alti: str):
            return alt1 if self._len == 1 else alti

        _type = {"int": "Integer", "bool": "Boolean", "float": "Real", "str": "String", "Enum": "Enumeration"}[
            self.typ.__qualname__
        ]  # translation of python to FMI primitives. Same for all components
        do_use_start = use_start(causality=self._causality, variability=self._variability, initial=self._initial)
        svars = []
        a_der = self.antiderivative()  # d a_der /dt = self, or None
        for i in range(self._len):
            sv = ET.Element(
                "ScalarVariable",
                {
                    "name": self.name + substr("", f"[{i}]"),
                    "valueReference": str(self.value_reference + i),
                    "description": "" if self.description is None else self.description,
                    "causality": self.causality.name,
                    "variability": self.variability.name,
                },
            )
            if isinstance(self._initial, Enum):
                sv.attrib.update({"initial": self._initial.name})
            if self._annotations is not None and i == 0:
                sv.append(ET.Element("annotations", self._annotations))
            #             if self.display is None or (self._len>1 and self.display[i] is None):
            #                 "display" = (self.unit, 1.0)

            # detailed variable definition
            info = ET.Element(_type)
            if do_use_start:  # a start value is to be used
                info.attrib.update({"start": self.fmi_type_str(self.start[i])})
            if _type in ("Real", "Integer", "Enumeration"):  # range to be specified
                xMin = self.range[i][0]
                if _type != "Real" or xMin > float("-inf"):
                    info.attrib.update({"min": str(xMin)})
                else:
                    info.attrib.update({"unbounded": "true"})
                xMax = self.range[i][1]
                if _type != "Real" or xMax < float("inf"):
                    info.attrib.update({"max": str(xMax)})
                else:
                    info.attrib.update({"unbounded": "true"})
            if _type == "Real":  # other attributes apply only to Real variables
                info.attrib.update({"unit": self.unit[i]})
                if self.display is not None and self.display[i] is not None and self.unit[i] != self.display[i][0]:
                    info.attrib.update({"displayUnit": self.display[i][0]})
                if a_der is not None:
                    info.attrib.update({"derivative": str(a_der.value_reference + i + 1)})

            sv.append(info)
            svars.append(sv)
        return svars

    def antiderivative(self) -> Variable | None:
        """Determine the variable which self is the derivative of.
        Return None if self is not a derivative.
        """
        if self.name.startswith("der(") and self.name.endswith(")"):
            return self.model.variable_by_name(self.name[4:-1])
        else:
            return None


# Utility functions for handling special variable types
def spherical_to_cartesian(vec: np.ndarray | tuple, deg: bool = False) -> np.ndarray:
    """Turn spherical vector 'vec' (defined according to ISO 80000-2 (r,polar,azimuth)) into cartesian coordinates."""
    if deg:
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


def cartesian_to_spherical(vec: np.ndarray | tuple, deg: bool = False) -> np.ndarray:
    """Turn the vector 'vec' given in cartesian coordinates into spherical coordinates.
    (defined according to ISO 80000-2, (r, polar, azimuth)).
    """
    r = np.linalg.norm(vec)
    if vec[0] == vec[1] == 0:
        if vec[2] == 0:
            return np.array((0, 0, 0), dtype="float")
        else:
            return np.array((r, 0, 0), dtype="float")
    elif deg:
        return np.array((r, degrees(acos(vec[2] / r)), degrees(atan2(vec[1], vec[0]))), dtype="float64")
    else:
        return np.array((r, acos(vec[2] / r), atan2(vec[1], vec[0])), dtype="float")


def cartesian_to_cylindrical(vec: np.ndarray | tuple, deg: bool = False) -> np.ndarray:
    """Turn the vector 'vec' given in cartesian coordinates into cylindrical coordinates.
    (defined according to ISO, (r, phi, z), with phi right-handed wrt. x-axis).
    """
    phi = atan2(vec[1], vec[0])
    if deg:
        phi = degrees(phi)
    return np.array((sqrt(vec[0] * vec[0] + vec[1] * vec[1]), phi, vec[2]), dtype="float")


def cylindrical_to_cartesian(vec: np.ndarray | tuple, deg: bool = False) -> np.ndarray:
    """Turn cylinder coordinate vector 'vec' (defined according to ISO (r,phi,z)) into cartesian coordinates.
    The angle phi is measured with respect to x-axis, right hand.
    """
    phi = radians(vec[1]) if deg else vec[1]
    return np.array((vec[0] * cos(phi), vec[0] * sin(phi), vec[2]), dtype="float")


def quantity_direction(quantity_direction: tuple, spherical: bool = False, deg: bool = False) -> np.ndarray:
    """Turn a 4-tuple, consisting of quantity (float) and a direction 3-vector to a direction 3-vector,
    where the norm denotes the direction and the length denotes the quantity.
    The return vector is always a cartesian vector.

    Args:
        quantity_direction (tuple): a 4-tuple consisting of the desired length of the resulting vector (in standard units (m or m/s))
           and the direction 3-vector (in standard units)
        spherical (bool)=False: Optional possibility to provide the input direction vector in spherical coordinates
        deg (bool)=False: Optional possibility to provide the input angle (of spherical coordinates) in degrees. Only relevant if spherical=True
    """
    if quantity_direction[0] < 1e-15:
        return np.array((0, 0, 0), dtype="float")
    if spherical:
        direction = spherical_to_cartesian(quantity_direction[1:], deg)  # turn to cartesian coordinates, if required
    else:
        direction = np.array(quantity_direction[1:], dtype="float")
    n = np.linalg.norm(direction)  # normalize
    return quantity_direction[0] / n * direction
