from __future__ import annotations

import xml.etree.ElementTree as ET  # noqa: N817
from enum import Enum, IntFlag
from math import acos, atan2, cos, degrees, radians, sin, sqrt
from typing import Callable, Type, TypeAlias

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


# Some special error classescls
class VariableInitError(Exception):
    """Special error indicating that something is wrong with the variable definition."""

    pass


class VariableRangeError(Exception):
    """Special Exception class signalling that a value is not within the range."""

    pass


class VariableUseError(Exception):
    """Special Exception class signalling that variable use was not in accordance with settings."""

    pass


class VarCheck(IntFlag):
    """Flags to denote how variables should be checked with respect to units and range. The aspects are indepent, but can be combined in the Enum through |.

    * none:     neither units nor ranges are expected or checked.
    * unitNone: only numbers without units expected when new values are provided.
      If units are provided during initialization, these should be base units (SE), i.e. unit and displayUnit are the same.
    * u_all:    expect always quantity and number and convert internally to base units (SE). Provide output as displayUnit
    * units:    flag to filter only on units, e.g ck & VarCheck.units
    * r_none:   no range is provided or checked
    * r_check:  range is provided and checked
    * ranges:  flag to filter on range, e.g. ck & VarCheck.ranges
    * all:     short for u_all | r_check
    """

    none = 0
    u_none = 0
    u_all = 1
    units = 1
    r_none = 0
    r_check = 2
    ranges = 2
    all = 2


class Variable(ScalarVariable):
    """Interface variable of an FMU. Can be a (python type) scalar variable. Extensions cover arrays (e.g. numpy array).
    The class extends pythonfmu.ScalarVariable, not using the detailed types (Real, Integer, ...), as these are handled internally.
    The recommended way to instantiate a Variable is through string values (with units for value0 and rng),
    but also dimensionless quantities and explicit class types are accepted.

    For proper understanding and usage the following should be noted:

    #. The Variable value is always owned by a model (see `self.model`).
    #. The current value of the variable is only directly accessible by the model through an attribute with the same name as `self.name`.
       Other access to the value is achieved through the `self.getter()` and the `self.setter( v)` + `self.on_set(v)` functions.
    #. It is recommended to register the Variable object as _`self.name` within the owner (model or sub-object of model),
       i.e. provide access as private object.
       In addition the model has access through the OrderedDict `.vars` ( {value_reference : variable object, ...})
    #. Compound variables (e.g. VariableNP), derived from Variable should always allow setting of single elements.
       Due to limitations in fmi2 all variables are translated to ScalarVariables.
       The setter function therefore adds changed variables to a dirty dict and uses on_set on the fully changed array.
       The on_set() method can be used to perform post-setting activities and are always performed on the whole 'vector'.
    #. _displayUnit is set to None if it is equal to _unit or if dimensionless units are used,
        as a quick signal that no conversion is needed.

    Args:
        model (obj): The model object where this variable relates to. Use model.add_variable( name, ...) to define variables
        name (str): Variable name, unique for whole FMU !!and registered in the model for direct value access!!
        description (str) = None: Optional description of variable
        causality (str) = 'parameter': The causality setting as string
        variability (str) = 'fixed': The variability setting as string
        initial (str) = None: Definition how the variable is initialized. Provide this explicitly if the default value is not suitable.
        typ (type)=None: The type of variable to expect as value0 and value. Since initial values are often set with strings (with units, see below), this is set explicitly.
           If None, _type is set to Enum/str if derived from these after disection or float if a number. 'int' is not automatically detected.
        value0 (PyType): The initial value of the variable.

           Optionally, the unit can be included, providing the initial value as string, evaluating to quantity of type typ a display unit and base unit.
           Note that the quantities are always converted to standard units of the same type, while the display unit may be different, i.e. the preferred user communication.
        rng (tuple) = (): Optional range of the variable in terms of a tuple of the same type as initial value. Can be specified with units (as string).

           * If an empty tuple is specified, the range is automatically determined. Note that it is thus not possible to automatically set single range elements (lower,upper) automatically
           * If None is specified, the initial value is chosen, i.e. no range. Applies to whole range tuple or to single elements (lower,upper)
           * For derived classes of Variable, the scalar ranges are in general calculated first and then used to specify derived ranges
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
        value0: PyType | None = None,
        rng: tuple = (),
        annotations: dict | None = None,
        value_check: VarCheck = VarCheck.all,
        fullInit: bool = True,
        getter: Callable | None = None,
        on_step: Callable | None = None,
        on_set: Callable | None = None,
    ):
        self.model = model
        self._causality, self._variability, self._initial = check_causality_variability_initial(
            causality, variability, initial
        )
        assert all(
            x is not None for x in (self._causality, self._variability, self._initial)
        ), f"The combination of causality {self._causality}, variability {self._variability}, initial {self._initial} is not allowed"
        super().__init__(name=name)  # the other properties are done here!
        self._annotations = annotations
        self._value_check = value_check
        self.on_step = on_step  # hook to define a function of currentTime and time step dT,
        # to be performed during Model.do_step for input variables
        self.on_set = on_set
        self._len = 1
        if getattr(self, "fullInit", True):  # stop the initialisation here (used by super-classes)
            self._value0, self._type, self.unit, self.displayUnit, self._range = self.value0_setter(value0, rng, typ)
            if getter is not None:
                self.getter = getter
            elif self._value_check == VarCheck.none and self.on_set is None:  # can use a simple getter/setter
                self.getter = lambda: getattr(self.model, self.local_name)
                self.setter = lambda v, idx: setattr(self.model, self.local_name, v)
            else:
                self.getter = self._getter
                self.setter = self._setter
            self.model.register_variable(self, self._value0)  # register in model and return index

    def __len__(self):
        return self._len  # This works also for derived variables, as long as _len is properly set

    # some getter and setter methods
    @property
    def type(self):
        return self._type

    @property
    def value0(self):
        return self._value0

    @property
    def causality(self) -> Causality:
        return self._causality

    @property
    def variability(self) -> Variability:
        return self._variability

    @property
    def initial(self) -> Initial:
        return self._initial

    @property
    def range(self):
        return self._range

    @range.setter
    def range(self, rng: tuple):
        """In some cases, e.g. when model paramters are changed, the range might change."""
        assert len(rng) == self._len, f"The new range {rng} should have length {self._len}"
        self._range = rng

    def value0_setter(self, val: PyType | None, rng: tuple, _type: Type | None):
        """Do checks (units, range, type) and set value0 during instantiation.
        The value0 is not meant to be changed!
        The value0 can be used to inform variable units. These may be explicitly used during the simulation run.
        The _type can be set explicitly or through val.
        """
        if _type == str:  # explicit free string
            return (val, _type, "", None, tuple())
        # if type is provided and no (initial) value. We set a default value of the correct type as 'example' value
        if val is None and _type is not None:
            if len(rng):
                val = rng[0]
            elif _type == Enum:
                val = 0
            elif _type == bool:
                val = False
            elif _type == int:
                val = 0
            elif _type == str:
                val = ""
            elif _type == float:
                val = 0.0
        #             else:
        #                 val = {float: 0.0, int: 0, bool: False, str: ""}[_type]
        assert val is not None, "Need a value at this stage"
        _value0, _unit, _displayUnit = self.disect_unit(val)  # first disect the initial value
        if _type is None:
            _type = self.auto_type(_value0)  # detect type from the _value0
        assert _type is not None, "Need a type at this stage"
        # .. but make still sure that the initial value is given as the correct typ
        #    (e.g. not providing an integer when a float was meant:
        try:
            _value0 = _type(_value0)
        except Exception:
            raise VariableInitError(f"conflicting combination of value0 {_value0} and type {_type}") from None
        _range = self.init_range(rng, _value0, _unit, _type) if VarCheck.r_check in self._value_check else None
        return (_value0, _type, _unit, _displayUnit, _range)

    def _getter(self) -> PyType:
        """Get variable value through model (which owns the value).
        This includes range checking and unit conversion as required in self._value_check.
        This works also for VariableNP! Vector elements can be retrieved by _getter()[idx].
        """
        val = getattr(self.model, self.local_name)  # get the raw current value from the model
        if self._causality == Causality.output:
            if VarCheck.r_check in self._value_check:
                assert self.check_range(
                    val
                ), f"Range violation in variable {self.name}, value {val}. Should be in range {self._range}"
            if VarCheck.u_all in self._value_check:  # provide output as displayUnit:
                val = self.unit_convert(val, tobase=False)
        return val

    def _setter(self, val: PyType | np.ndarray | tuple, idx: int | None = None):
        """Set variable value through model (which owns the value).
        This includes range and type checking + unit conversion as required in self._value_check.
        For these singel-valued variables on_set is run immediatelly (if set). Not registered in model._dirty'.
        """
        assert not isinstance(val, (np.ndarray, tuple)), f"Calling Variable._setter with value {val}"
        if VarCheck.u_all in self._value_check:  # expect quantity as displayUnit and convert to base units (SE)
            val = self.unit_convert(val)
        # range is provided and checked:
        if VarCheck.r_check in self._value_check:
            assert self.check_range(
                val
            ), f"Range violation in variable {self.name}, value {val}. Should be in range {self._range}"
        setattr(self.model, self.local_name, val if self.on_set is None else self.on_set(val))  # model is owner!

    def auto_type(self, exampleVal: PyType):
        """Determine the Variable type from a provided example value.
        Since variables can be initialized using strings with units, the type can only be determined when the value is disected.
        Moreover, the value may indicate an integer, while the variable is designed a float. Therefore int Variables must be explicitly specified.
        """
        if isinstance(exampleVal, bool):
            return bool
        elif isinstance(exampleVal, (int, float)):
            return float
        else:
            return type(exampleVal)

    def check_value(
        self,
        val: PyType,
        _value0: PyType | None = None,
        _range: tuple | None = None,
        _displayUnit: str | None = None,
    ):
        """Check a provided value and return the quantity, unit, displayUnit and range. Processing like on_set is not performed here.
        Note: The function works also for components of derived variables if all input parameters are provided explicitly.
        Note 2: Variable initialization is not handled here.
        .. todo:: better coverage with respect to variability and causality on when it is allowed to change the value.

        Args:
            val: the raw value to be checked. May be a string with units (to be disected as part of this function)
            _value0 (int,float,...)=None: the processed value0 (unit disected and range checked). self._value0 if None and not initial
            _range (tuple)=None: the range provided as tuple. self._range if None and not initial
            _displayUnit (str)=None: the displayUnit (unit as which the variable is expected). self.displayUnit if None and not initial
        Returns:
            val
        """
        if self._variability == Variability.constant:
            raise VariableUseError(f"It is not allowed to change the value of variable {self.name}")
        if VarCheck.u_all in self._value_check:
            val, _unit, du = self.disect_unit(val)  # first disect the initial value
            if _displayUnit is None:
                _displayUnit = self.displayUnit
            if du != _displayUnit:
                raise VariableUseError(f"The expected unit of variable {self.name} is {_displayUnit}. Got {du}")
        if VarCheck.r_check in self._value_check:  # range checking is performed
            if _value0 is None:
                _value0 = self._value0
            if _range is None:
                _range = self._range
            assert self.check_range(
                val, rng=_range
            ), f"The value {str(val)} is not accepted within variable {self.name}. Range is set to {str(_range)}"
        return val

    def __str__(self):
        return f"Variable {self.name}. Initial: {str(self._value0)}. Current {self.getter()} [{self.unit}]."

    @classmethod
    def _get_auto_extreme(cls, var: PyType):
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
            raise VariableInitError(
                f"Unlimited integer variables do not make sense in Python. Please provide explicit limits for variable {cls.name} or set the type to float."
            )
        elif isinstance(var, Enum):
            return (min(x.value for x in type(var)), max(x.value for x in type(var)))
        else:
            return tuple()  # return an empty tuple (no range specified, e.g. for str)

    def init_range(self, rng: tuple, value0: PyType | None = None, unit: str | None = None, _type: Type | None = None):
        """Initialize the variable range.
        Function can be called separately per component by derived classes.
        The value0 and unit can be explicitly provided, or self._* is used, if None.
        """
        if value0 is None:
            value0 = self._value0
        if unit is None:
            unit = self.unit
        if _type is None:
            _type = self.type
        # === initialize the variable range
        if rng is None:  # set a zero-interval range.
            # Note: this makes only sense for combined variables, where some components may be fixed and others variable
            _range = (value0, value0)
        elif not len(rng):  # empty tuple => automatic range (for float and Enum)
            _range = self._get_auto_extreme(_type(value0))
        else:
            assert len(rng) == 2, f"Range specification must be a tuple of length 2. Found {rng}"
            _rng = []
            for r in rng:
                if r is None:  # no range => fixed to initial value
                    q = value0
                else:
                    q, u, du = self.disect_unit(r)
                    if (
                        (q == 0 or q == float("inf") or q == float("-inf")) and u == "dimensionless"
                    ):  # we accept that no explicit unit is supplied when the quantity is 0 or inf
                        u = unit
                    elif len(unit) and len(u) and unit != u:
                        raise VariableInitError(
                            f"The supplied range value {str(r)} does not conform to the unit type {unit}"
                        )
                _rng.append(q)
            try:  # check variable type
                _rng = [_type(x) for x in _rng]  # type: ignore
            except Exception as err:
                raise VariableRangeError(
                    f"The given range {rng} and type {_type} is not compatible with the provided value {value0}"
                ) from err
            if all(isinstance(x, _type) for x in _rng):
                _range = tuple(_rng)
            else:
                raise NotImplementedError("What else?")
        if not self.check_range(value0, _range, _type):  # checks on initial value
            raise VariableInitError(f"The provided value {value0} is not in the valid range {_range}")
        return _range

    def check_range(
        self, val: PyType | np.ndarray | tuple, rng: tuple | None = None, _type: Type | None = None
    ) -> bool:
        """Check the provided 'val' with respect to the variable range.

        Args:
            val (PyType): the value to check.

               * If tuple, check whether the provided range tuple is adequate with respect to variable type.
               * It is allowed to replace a value within the tuple by None, but only one value is not allowed
               * If None, the range is set equal to the initalVal, which makes only sense if the Variable is a component of a derived Variable, where other components are not constants
               * If not tuple or None, check with respect to variable type and whether it is within range.
            rng (tuple,None)=None: optional explicit specification of target range. If None, use self.range
        Returns:
            True/False with respect to whether val is the right type and is within range. self._range is registered as side-effect
        """
        if rng is None:
            rng = self.range
        if _type is None:
            _type = self.type
        if not isinstance(val, (bool, int, float, str, Enum)) and len(val):  # go per component
            return all(self.check_range(val[i], self.range[i]) for i in range(len(self)))
        else:  # single component check
            if hasattr(self, "_type") and _type != type(val):
                try:
                    val = _type(val)  # try to cast the value
                except Exception:  # give up
                    return False
            if _type == bool:
                return isinstance(val, bool)
            elif isinstance(val, Enum):
                return isinstance(val, _type)
            elif isinstance(val, str):
                return True  # no further requirements for str
            elif isinstance(val, (int, float)) and all(isinstance(x, (int, float)) for x in rng):
                return rng is None or rng[0] <= val <= rng[1]  # type: ignore
            else:
                raise AssertionError(f"Unhandled combination of val {val} and range {rng} in check_range") from None

    def fmi_type_str(self, val: PyType) -> str:
        """Translate the provided type to a proper fmi type and return as string.
        See types defined in schema fmi2Unit.xsd.
        """
        if self._type == bool:
            return "true" if val else "false"
        else:
            return str(val)

    def xml_scalarvariables(self, modelvariables: ET.Element):
        """Generate <ScalarVariable> XML code with respect to this variable and append to modelvariables.
        For compound variables, all elements are appended.
        Note that ScalarVariable attributes should all be listed in __attrs dictionary.
        Since we do not use the derived classes Real, ... we need to generate the detailed variable definitions here.
        The following attributes are so far not supported: declaredType, derivative, reinit.
        Nothing returned. Elements are appended to the provided <ModelVariables> element.

        Args:
            modelvariables (ET.Element): the <ModelVariables> root element to which <ScalarVariables are appended.

            typ: the type can be explicitly provided (for derived variables), otherwise self._type is used
            value0: a value0 can be explicitly provided (for components of derived variables)
            valueReference: For compound variables this is provided explicitly for >0 elements.
            range: a range tuple can be explicitly provided (for components of derived variables)
            unit: a unit can be explicitly provided (for components of derived variables)
            displayUnit: a displayUnit can be explicitly provided (for components of derived variables)

        """

        def substr(alt1: str, alti: str):
            return alt1 if self._len == 1 else alti

        def sub(obj, i: int):
            if isinstance(self, VariableNP):
                return obj[i]
            else:
                return obj

        declaredType = {"int": "Integer", "bool": "Boolean", "float": "Real", "str": "String", "Enum": "Enumeration"}[
            self.type.__qualname__
        ]  # translation of python to FMI primitives. Same for all components
        do_use_start = use_start(self._causality, self._variability, self._initial)
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
            #             if self.displayUnit is None or (self._len>1 and self.displayUnit[i] is None):
            #                 "displayUnit" = (self.unit, 1.0)

            # detailed variable definition
            varInfo = ET.Element(declaredType)
            if do_use_start:  # a start value is to be used
                varInfo.attrib.update({"start": self.fmi_type_str(sub(self.value0, i))})
            if declaredType in ("Real", "Integer", "Enumeration"):  # range to be specified
                xMin = sub(self.range, i)[0]
                if declaredType != "Real" or xMin > float("-inf"):
                    varInfo.attrib.update({"min": str(xMin)})
                else:
                    varInfo.attrib.update({"unbounded": "true"})
                xMax = sub(self.range, i)[1]
                if declaredType != "Real" or xMax < float("inf"):
                    varInfo.attrib.update({"max": str(xMax)})
                else:
                    varInfo.attrib.update({"unbounded": "true"})
            if declaredType == "Real":  # other attributes apply only to Real variables
                varInfo.attrib.update({"unit": sub(self.unit, i)})
                if self.displayUnit is not None and sub(self.displayUnit, i) is not None:
                    if sub(self.unit, i) != sub(self.displayUnit, i)[0]:
                        varInfo.attrib.update({"displayUnit": sub(self.displayUnit, i)[0]})
            sv.append(varInfo)

            modelvariables.append(sv)

    def disect_unit(self, quantity: PyType) -> tuple[PyType, str, tuple | None]:
        """Disect the provided quantity in terms of magnitude and unit, if provided as string.
        If another type is provided, dimensionless units are assumed.

        Args:
            quantity (PyType): the quantity to disect. Should be provided as string, but also the trivial cases (int,float,Enum) are allowed.
            A free string should not be used and leads to a warning
        Returns:
            the magnitude in base units, the base unit and the unit as given together with the conversion factor (from displayUnit to baseUnit)
        """
        assert isinstance(
            quantity, (str, int, float, bool, Enum)
        ), f"Wrong value type {str(type(quantity))} for scalar variable {self.name}. Found {quantity}."
        if isinstance(quantity, str):  # only string variable make sense to disect
            try:
                q = self.model.ureg(quantity)  # parse the quantity-unit and return a Pint Quantity object
                if isinstance(q, (int, float)):
                    return q, "", None  # integer or float variable with no units provided
                elif isinstance(q, Quantity):  # pint.Quantity object
                    displayUnit = str(q.units)
                    qB = (
                        q.to_base_units()
                    )  # transform to base units ('SI' units). All internal calculations will be performed with these
                    if isinstance(self, VariableNP):
                        val = np.array(qB.magnitude, self._type).tolist()
                    else:
                        val = qB.magnitude  # Note: numeric types are not converted, e.g. int to float
                    uB = str(qB.units)
                    if displayUnit == uB:
                        return (val, uB, None)
                    else:  # explicit displayUnits necessary
                        if val == 0:  # not suitable to identify the conversion factor
                            return (
                                val,
                                uB,
                                (displayUnit, self.model.ureg.Quantity(1.0, displayUnit).to_base_units().magnitude),
                            )
                        else:
                            return val, uB, (displayUnit, val / q.magnitude)
                else:
                    raise VariableInitError(f"Unknown quantity {quantity} to disect") from None
            # no recognized units. Assume a free string. ??Maybe we should be more selective about the exact error type:
            except Exception as warn:
                logger.warning(
                    f"The string quantity {quantity} could not be disected: {warn}. If it is a free string and explicit type 'typ=str' should be provided to avoid this warning"
                )
                return (quantity, "", None)
        else:
            return (quantity, "dimensionless", None)

    def unit_convert(self, val: PyType | np.ndarray | tuple, idx: int | None = None, tobase: bool | float = True):
        """Convert the value 'val' between displayUnit and baseUnit.

        Args:
            val: the value to be converted
            idx: Optional possibility to specify the component of a compound variable
            tobase=True: convert from displayUnit to baseUnit, False: from baseUnit to displayUnit, float: use the given factor

        self.displayUnit is either None (no conversions) or a tuple ( name of displayUnit, conversion factor tobase)
        """
        if isinstance(val, str):  # single value provided as string with units
            val, bU, du = self.disect_unit(val)
            if (idx is None and bU != self.unit) or (idx is not None and bU != self.unit[idx]):
                raise ValueError(f"The unit {bU} is not allowed for variable {self.name}")
            if tobase:
                return val
        # val is numeric
        if isinstance(val, (np.ndarray, tuple)):
            if idx is None:  # the whole variable
                value = [self.unit_convert(val[i], idx=i, tobase=tobase) for i in range(len(self))]
                return np.array(value, dtype=self._type)
            else:  # an element of the array
                return self.unit_convert(val[idx], idx=idx, tobase=tobase)
        else:  # val is a single numeric value
            if idx is None or self.displayUnit is None:  # no indexing and no conversion required
                return val
            elif (
                idx is not None and self.displayUnit[idx] is None
            ):  # indexing and no conversion (for this index) required
                return val
            else:
                if idx is None:
                    fac = self.displayUnit[1]
                else:
                    fac = self.displayUnit[idx][1]
                assert isinstance(fac, float), f"Conversion factor must be a float. Found {fac}"
                assert isinstance(val, (int, float)), f"Value to convert must be int or float. Found {val}"
                return val * fac if tobase else val / fac


class VariableNP(Variable):
    """NumPy array variable as extension of Variable.
    The variable is internally kept as one object (with arrays of values, ranges, ...) and only when generating e.g. an FMU, the variable is split.

        Args:
            model (obj): The model object where this variable relates to. Use model.add_variable( name, ...) to define variables
            name (str): Variable name, unique for whole FMU. The array components get names <name>[0],...
            description (str) = None: Optional description of variable. Array components get empty descriptions
            causality (str) = 'parameter': The causality setting as string. Same for whole array
            variability (str) = 'fixed': The variability setting as string. Same for whole array
            initial (str) = None: Definition how the variable is initialized. Provide this explicitly if the default value is not suitable. Same for the whole array
            value0 (tuple) = (): The initial value of the array components.

               This determines also the variable type of scalars in terms of Python primitive types.
            unit (Unit, tuple) = '': Optional unit of variable(s). If only one value is provided all components are assumed equal.
            rng (tuple) = (): Optional range of the array components. If only one tuple is provided, all components are assumed equal
            annotations (dict) = None: Optional variable annotations provided as dict
            value_check (bool) = True: Optional possibility to bypass checking of new values with repect to type and range. Same for whole array
            typ (np.dtype) = None: Optional possibility to explicitly set the np.array type. Default: float64
            on_step (callable) = None: Optonal possibility to register a function of (currentTime, dT) to be run during Model.do_step,
               e.g. if the variable represents a speed, the object can be translated speed*dT, if |speed|>0
            on_set (callable) = None: Optional possibility to specify a function of (newVal) to be run when the variable is changed.
               This is useful for conditioning of input variables, so that calculations can be done once after a value is changed and do not need to be repeated on every simulation step.
               If given, the function shall apply to a value as expected by the variable (e.g. if there are components) and after unit conversion.
    """

    def __init__(
        self,
        model,
        name: str,
        description: str = "",
        causality: str | None = "parameter",
        variability: str | None = "fixed",
        initial: str | None = None,
        value0: tuple = (),
        unit: str | tuple = "",
        rng: tuple = tuple(),
        annotations: dict | None = None,
        value_check=VarCheck.all,
        typ=None,
        getter: Callable | None = None,
        on_step: Callable | None = None,
        on_set: Callable | None = lambda v: v,
    ):
        self.fullInit = False  # when calling super, the initialization is stopped where the array becomes relevant
        super().__init__(
            model=model,
            name=name,
            description=description,
            causality=causality,
            variability=variability,
            initial=initial,
            annotations=annotations,
            value_check=value_check,
            on_step=on_step,
            on_set=on_set,
        )  # do basic initialization
        self._type = np.float64 if typ is None or not isinstance(typ, np.dtype) else typ
        self._len = len(value0)
        if not len(rng):
            rng = ((),) * self._len
        _value0, _unit, _displayUnit, _range = [], [], [], []
        for i in range(len(value0)):
            _i, _u, _d = self.disect_unit(value0[i])  # first disect the initial value
            _r = self.init_range(rng[i], _i, _u, _type=self.type) if VarCheck.r_check in self._value_check else None
            _value0.append(_i)
            _unit.append(_u)
            _displayUnit.append(_d)
            _range.append(_r)
        self._value0 = np.array(_value0, self.type)
        self.unit = tuple(_unit)
        if all(d is None for d in _displayUnit):
            self.displayUnit = None  # a quick way to state that no unit transformation is needed on the whole variable
        else:
            self.displayUnit = tuple(_displayUnit)
        self._range = tuple(_range)
        self._dirty = False
        self.getter = self._getter if getter is None else getter  # type: ignore
        self.setter = self._setter
        self.model.register_variable(self, self._value0)  # register in model and set the current value
        # Note: Only the first value_reference contains the variable

    @property
    def type(self):
        return type(np.array(0, dtype=self._type).tolist())  # translate to the native python type

    def _setter(self, val: PyType | np.ndarray | tuple, idx: int | None = None):
        """Set variable value through model (which owns the value).
        This includes range and type checking + unit conversion as required in self._value_check.
        We cannot set the value itself just here, because partial array changes can create problems.
        Therefore the changes are collected in the 'dirty' dict of the model
        and are effectuated as first action of 'do_step', including on_set().
        """
        if hasattr(val, "__iter__"):  # the whole whole array
            assert isinstance(val, (np.ndarray, tuple, list)), f"Erroneous VariableNP value {val} in {self.name}"
            assert len(self) == len(val), f"Erroneous dimension in {val} of variable {self.name}. Expected {len(self)}"
            if VarCheck.u_all in self._value_check:  # expect quantity as displayUnit and convert to base units (SE)
                for i in range(self._len):
                    val[i] = self.unit_convert(val[i], i)  # type: ignore
            if VarCheck.r_check in self._value_check:  # range is provided and checked:
                for i in range(self._len):
                    msg = f"Range violation in variable {self.name}[{i}], value {val[i]}. Should be in {self._range[i]}"
                    assert self.check_range(val[i], rng=self._range[i]), msg
            self.model.ensure_dirty(self, np.array(val, dtype=self._type))  # register the new value
        #            setattr(self.model, self.local_name, val) # model is owner!

        else:
            assert idx is not None, "Integer idx needed in this case"
            assert 0 <= idx < self._len, f"Erroneous index {idx} in VariableNP {self.name}"
            msg = f"Erroneous VariableNP value element {val}, index {idx} in {self.name}"
            assert isinstance(val, (str, int, float, bool)), msg
            if VarCheck.u_all in self._value_check:  # expect quantity as displayUnit and convert to base units (SE)
                val = self.unit_convert(val)
            if VarCheck.r_check in self._value_check:  # range is provided and checked:
                msg = f"Range violation in variable {self.name}[{idx}], value {val}. Should be in {self._range[idx]}"
                assert self.check_range(val, rng=self._range[idx]), msg
            self.model.ensure_dirty(self, val, idx)


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
            return np.array((0, 0, 0), dtype="float64")
        else:
            return np.array((r, 0, 0), dtype="float64")
    elif asDeg:
        return np.array((r, degrees(acos(vec[2] / r)), degrees(atan2(vec[1], vec[0]))), dtype="float64")
    else:
        return np.array((r, acos(vec[2] / r), atan2(vec[1], vec[0])), dtype="float64")


def cartesian_to_cylindrical(vec: np.ndarray | tuple, asDeg: bool = False) -> np.ndarray:
    """Turn the vector 'vec' given in cartesian coordinates into cylindrical coordinates.
    (defined according to ISO, (r, phi, z), with phi right-handed wrt. x-axis).
    """
    phi = atan2(vec[1], vec[0])
    if asDeg:
        phi = degrees(phi)
    return np.array((sqrt(vec[0] * vec[0] + vec[1] * vec[1]), phi, vec[2]), dtype="float64")


def cylindrical_to_cartesian(vec: np.ndarray | tuple, asDeg: bool = False) -> np.ndarray:
    """Turn cylinder coordinate vector 'vec' (defined according to ISO (r,phi,z)) into cartesian coordinates.
    The angle phi is measured with respect to x-axis, right hand.
    """
    phi = radians(vec[1]) if asDeg else vec[1]
    return np.array((vec[0] * cos(phi), vec[0] * sin(phi), vec[2]), dtype="float64")


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
        return np.array((0, 0, 0), dtype="float64")
    if asSpherical:
        direction = spherical_to_cartesian(quantityDirection[1:], asDeg)  # turn to cartesian coordinates, if required
    else:
        direction = np.array(quantityDirection[1:], dtype="float64")
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
        if length == 1:  # a scalar
            _var = Variable(
                model,
                base,
                description=var.attrib["description"],
                causality=_causality,
                variability=_variability,
                initial=_initial,
                typ=_typ,
                value0=var[0].attrib.get("start", None),
                rng=range_from_fmu(var[0]),
            )

        else:  # an array
            _var = VariableNP(
                model,
                base,
                description=var.attrib["description"],
                causality=_causality,
                variability=_variability,
                initial=_initial,
                typ=_typ,
                value0=tuple(v[0].attrib.get("start", None) for v in el[idx : idx + length]),
                rng=tuple(range_from_fmu(v[0]) for v in el[idx : idx + length]),
            )
        idx += length
        if idx >= len(el):
            break
