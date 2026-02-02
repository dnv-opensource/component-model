from __future__ import annotations

import logging
import xml.etree.ElementTree as ET  # noqa: N817
from enum import Enum
from typing import Any, Callable, Never, Sequence, TypeAlias

import numpy as np
from pythonfmu.enums import Fmi2Causality as Causality
from pythonfmu.enums import Fmi2Initial as Initial
from pythonfmu.enums import Fmi2Variability as Variability
from pythonfmu.variables import ScalarVariable

from component_model.enums import Check, check_causality_variability_initial, use_start
from component_model.range import Range
from component_model.unit import Unit
from component_model.variable_naming import ParsedVariable

logger = logging.getLogger(__name__)
PyType: TypeAlias = str | int | float | bool | Enum
# RngSingle: TypeAlias = tuple[int | float | None, int | float | None] | None | tuple[()]
RngSingle: TypeAlias = tuple[Any, Any] | None | Sequence[Never]
Numeric: TypeAlias = int | float
Compound: TypeAlias = tuple[PyType, ...] | list[PyType] | np.ndarray


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
    #. The current value of the variable is directly accessible through the owner.
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
        start (PyType): The initial value(s) of the variable.

           Optionally, the unit can be included, providing the initial value as string,
           evaluating to quantity of type typ a display unit and base unit.
           Note that the quantities are always converted to standard units of the same type, while the display unit may be different,
           i.e. the preferred user communication.
        rng ((tuple of) range spec) = (): Optional range of the variable in terms of a (tuple of) range spec(s).
           Should be specified with units (as string). See Range class for details.

           * If an empty tuple is specified, the range is automatically determined.
             That is only possible for float or enum type variables, where the former evaluates to (-inf, inf).
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
           The function is invisible by the user specifying inputs to the variable.
        owner = None: Optional possibility to overwrite the default owner.
           If the related model uses structured variable naming this should not be necessary,
           but for flat variable naming within complex models (not recommended) ownership setting might be necessary.
        local_name (str) = None: Optional possibility to overwrite the automatic determination of local_name,
           which is used to access the variable value and must be a property of owner.
           This is convenient for example to link a derivative name to a variable if the default name,
           i.e. der_<base-var> is not acceptable.
    """

    def __init__(
        self,
        model: Any,
        name: str,
        description: str = "",
        causality: str | None = "parameter",
        variability: str | None = "fixed",
        initial: str | None = None,
        typ: type | None = None,
        start: PyType | Compound | None = None,
        rng: RngSingle | tuple[RngSingle, ...] = tuple(),
        annotations: dict[str, Any] | None = None,
        value_check: Check = Check.all,
        on_step: Callable[[float, float], None] | None = None,
        on_set: Callable[[int | float | np.ndarray], int | float | np.ndarray] | None = None,
        owner: Any | None = None,
        local_name: str | None = None,
    ):
        from component_model.model import Model

        assert isinstance(model, Model)
        self.model = model
        self._causality, self._variability, self._initial = check_causality_variability_initial(
            causality, variability, initial
        )
        assert all(x is not None for x in (self._causality, self._variability)), (
            f"Combination causality {self._causality}, variability {self._variability}, initial {self._initial} is not allowed"
        )
        super().__init__(name=name, description=description, getter=self.getter, setter=self.setter)

        parsed = ParsedVariable(name, self.model.variable_naming)

        self._annotations = annotations
        self._check = value_check  # unique for all elements in compound variables
        self._typ: type | None = typ  # preliminary. Will be adapted if not explicitly provided (None)

        self.on_step = on_step  # hook to define a function of currentTime and time step dT,
        # to be performed during Model.do_step for input variables
        self.on_set = on_set
        # Note: the _len is a central property, distinguishing scalar and compound variables.
        self._unit: tuple[Unit, ...]
        self._start: tuple[PyType, ...] = tuple()

        if owner is None:
            oh = self.model.owner_hierarchy(parsed.parent)
            if owner is None:
                self.owner = oh[-1]
        else:
            self.owner = owner
        basevar: Variable | None = None
        if local_name is None:
            if parsed.der > 0:  # is a derivative of 'var'
                self.local_name = f"der{parsed.der}_{parsed.var}"
                if not hasattr(self.owner, self.local_name):  # a virtual derivative
                    basevar = self.model.add_derivative(
                        self.name, parsed.as_string(("parent", "var", "der"), primitive=True)
                    )
                    assert isinstance(basevar, Variable), f"The primitive of {self.name} must be a Variable object"
                    assert basevar.typ is float, f"The primitive of {self.name} shall be float. Found {basevar.typ}"
                    self._typ = float
                    if self.on_step is None:
                        self.on_step = self.der1
            else:
                self.local_name = parsed.var
        else:
            self.local_name = local_name  # use explicitly provided local name

        if start is None:
            assert local_name is None, f"{self.name} Default start value only defined for derivatives"
            assert basevar is not None, f"{self.name} basevar needed at this point"
            self._start, self._unit = Unit.derivative(basevar.unit)
        elif self._typ is str:
            assert isinstance(start, str), f"Scalar str expected. Found {start}"
            self._start, self._unit = (start,), (Unit(None),)  # explicit free string variable
        elif not isinstance(start, (tuple, list, np.ndarray)):
            self._start, self._unit = Unit.make(start, no_unit=False)
        else:
            self._start, self._unit = Unit.make_tuple(start, no_unit=False)
        self._len = 1 if self._typ is str else len(self._start)
        if self._typ is None:  # try to adapt using _start and _unit
            self._typ = self.auto_type(self._start, self._unit)
        assert isinstance(self._typ, type)
        if self._typ is not Enum:  # Enums are already checked and casting does not work
            self._start = tuple([self._typ(s) for s in self._start])  # make sure that python type is correct

        ck = Range.is_valid_spec(rng, self._len, self._typ)
        if ck != 0:
            raise ValueError(f"{self.name} invalid range spec '{rng}'. Error: {Range.err_code_msg(ck)}") from None
        self._range: tuple[Range, ...]
        # Defining the _range. self._start is in base units, while rng is in display units
        if self._typ is str:  # explicit free string. String arrays are so far not implemented
            assert isinstance(start, str)
            self._range = (Range(self._start[0], unit=self._unit[0]),)  # Strings have fixed range
        else:
            if self._len == 1:
                self._range = (Range(self._start[0], rng, self._unit[0]),)  # type: ignore[arg-type]  ## is_valid_spec
            else:
                _rng: list[Range] = []
                for i in range(self._len):
                    if rng is None or not len(rng):
                        _rng.append(Range(self._start[i], rng, self._unit[i]))  # type: ignore[arg-type]
                    else:
                        _rng.append(Range(self._start[i], rng[i], self._unit[i]))
                self._range = tuple(_rng)

        if not self.check_range(self._start, disp=False):  # range checks of initial value
            logger.critical(f"The provided value {self._start} is not in the valid range {self._range}")
            raise ValueError(f"The provided value {self._start} is not in the valid range {self._range}")
        self.model.register_variable(self)
        assert len(self._start) > 0, "Empty tuples are not handled here:"
        try:
            setattr(self.owner, self.local_name, np.array(self._start, self.typ) if self._len > 1 else self._start[0])
        except AttributeError as _:  # can happen if a @property is defined for local_name, but no @local_name.setter
            pass

    def der1(self, current_time: float, step_size: float):
        """Ramp the base variable value up or down within step_size."""
        der = np.array(getattr(self.owner, self.local_name))  # the current slope value
        if not np.allclose(der, 0.0):
            basevar = self.model.derivatives[self.name]  # base variable object
            val = np.array(getattr(self.owner, basevar.local_name))  # previous value of base variable
            newval = val + step_size * der
            basevar.setter_internal(newval, -1)  # , True)

    #     def _parse_start( start:None|PyType|tuple[PyType])-> tuple[PyType|np.ndarray,):
    #         """Read start value(s), extract unit(s) and return everything needed for the simulation."""
    #         if start is None:
    #             assert local_name is None, f"{self.name} Default start value only defined for derivatives"
    #             assert basevar is not None, f"{self.name} basevar needed at this point"
    #             self._start, self._unit = Unit.derivative(basevar.unit)
    #         elif self._typ is str or self._typ is Enum:
    #
    #         not isinstance(start, tuple)):
    #             self._start, self._unit = Unit.make(start, no_unit=True)  # type: ignore  ## type of start should be ok
    #         else:
    #             self._start, self._unit = Unit.make_tuple(start, no_unit=False)
    #         self._len = 1 if self._typ is str else len(self._start)
    #         if self._typ is None:  # try to adapt using start
    #             self._typ = self.auto_type(self._start)
    #         assert isinstance(self._typ, type)
    #         self._start = tuple([self._typ(s) for s in self._start])  # make sure that python type is correct

    # disable super() functions and properties which are not in use here
    def to_xml(self) -> ET.Element:
        logger.critical("The function to_xml() shall not be used from component-model")
        raise NotImplementedError("The function to_xml() shall not be used from component-model") from None

    # External access to read-only variables:
    def __len__(self) -> int:
        return self._len  # This works also compound variables, as long as _len is properly set

    @property
    def start(self) -> tuple[PyType, ...]:
        return self._start

    @property
    def unit(self):
        """Get the unit object."""
        return self._unit

    @property
    def range(self):
        return self._range

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
        assert self._typ is not None, "Need a proper type at this stage"
        assert isinstance(values, (Sequence, np.ndarray)), "A sequence is expected as values"
        if idx == -1 and self._len == 0:  # the whole scalar
            idx = 0

        if issubclass(self._typ, Enum):  # Enum types may be supplied as int. Convert
            for i in range(self._len):
                if isinstance(values[i], int):
                    values[i] = self._typ(values[i])  # type: ignore

        if self._check & Check.ranges:  # do that before unit conversion, since range is stored in display units!
            if not self.check_range(values, idx):
                logger.error(f"set(): values {values} outside range.")

        if self._check & Check.units:  #'values' expected as displayUnit. Convert to unit
            if idx >= 0:  # explicit index of single values
                dvals = [self._unit[idx].to_base(values[0])]  # type: ignore
            else:  # the whole array
                dvals = []
                for i in range(self._len):
                    if values[i] is None:  # keep the value
                        dvals.append(getattr(self.owner, self.local_name)[i])
                    else:
                        dvals.append(self._unit[i].to_base(values[i]))
        else:  # no unit issues
            if self._len == 1:
                dvals = [values[0] if values[0] is not None else getattr(self.owner, self.local_name)]
            else:
                dvals = [
                    values[i] if values[i] is not None else getattr(self.owner, self.local_name)[i]
                    for i in range(self._len)
                ]
        self.setter_internal(dvals, idx)  # do the setting, or flag as dirty

    def setter_internal(
        self,
        values: Sequence[int | float | bool | str | Enum | None] | np.ndarray,
        idx: int = -1,
    ):
        """Do internal setting of values (no range checking and units expected internal), including dirty flags."""
        if self._len == 1:
            try:
                _val = values[0]
            except IndexError:  # Exception as err:
                _val = values
            setattr(self.owner, self.local_name, _val if self.on_set is None else self.on_set(_val))  # type: ignore
        elif idx >= 0:
            if values[0] is not None:  # Note: only the indexed value is provided, as list!
                val = getattr(self.owner, self.local_name)
                val[idx] = values[0]
                setattr(self.owner, self.local_name, val)
                if self.on_set is not None:
                    self.model.dirty_ensure(self)
        else:  # the whole array
            arr: np.ndarray = np.array(values, self._typ)
            setattr(self.owner, self.local_name, arr if self.on_set is None else self.on_set(arr))
        if self.on_set is None:
            logger.debug(f"SETTER {self.name}, {values}[{idx}] => {getattr(self.owner, self.local_name)}")

    def getter(self) -> list[PyType]:
        """Get the value (output a value from the model), including range checking and unit conversion.
        For compound variables, the whole variable is returned as list (even for scalar variables).
        Returned value lists can later be indexed/sliced to get elements of (compound) variables.
        """
        assert self._typ is not None, "Need a proper type at this stage"
        if self._len == 1:
            value = getattr(self.owner, self.local_name)  # work with the single value
            if issubclass(self._typ, Enum):  # native Enums do not exist in FMI2. Convert to int
                values = [value.value]
            else:
                if not isinstance(value, self._typ):  # other type conversion
                    value = self._typ(value)  # type: ignore[call-arg] ## only mypy
                if self._check & Check.units:  # Convert 'value' base unit -> display.u
                    values = [self._unit[0].from_base(value)]
                else:
                    values = [value]

        else:  # compound variable
            values = list(getattr(self.owner, self.local_name))  # make value available as copy
            if issubclass(self._typ, Enum):  # native Enums do not exist in FMI2. Convert to int
                for i in range(self._len):
                    values[i] = values[i].value
            else:
                for i in range(self._len):  # check whether conversion to _typ is necessary
                    if not isinstance(values[i], self._typ):
                        values[i] = self._typ(values[i])  # type: ignore[call-arg]  ## only mypy
            if self._check & Check.units:  # Convert 'value' base unit -> display.u
                for i in range(self._len):
                    values[i] = self._unit[i].from_base(values[i])

        if self._check & Check.ranges and not self.check_range(values, -1):  # check the range if so instructed
            logger.error(f"getter(): Value of {self.name}: {values} outside range {self.range}!")
        return values

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
            ck = self._typ is str
        elif self._len > 1 and idx < 0:  # check all components
            assert isinstance(values, (Sequence, np.ndarray)) and len(values) == self._len, (
                f"Values {values} not sufficient. Need {self._len}"
            )
            ck = all(r.check(v, self._typ, u, disp) for v, r, u in zip(values, self._range, self._unit, strict=True))
        else:
            ck = self._range[idx].check(values[0], self._typ, self._unit[idx], disp)
        return ck

    def fmi_type_str(self, val: PyType) -> str:
        """Translate the provided type to a proper fmi type and return it as string.
        See types defined in schema fmi2Unit.xsd.
        """
        if self._typ is bool:
            return "true" if val else "false"
        else:
            return str(val)

    @classmethod
    def auto_type(cls, vals: tuple[PyType, ...], units: tuple[Unit, ...]) -> type:
        """Determine the Variable type from a set of example values and related Unit objects.

        Variable type must be unique for the whole set of vals/units.
        Since variables can be initialized using strings with units,
        the type can only be determined when the value is disected and the units defined.
        Moreover, the value may indicate an integer, while the variable is designed a float.
        int type is therefore only decided if all vals are int and if no unit is disected.
        """
        types: list[type] = []
        for v, u in zip(vals, units, strict=True):
            types.append(type(v))
            if isinstance(v, (bool, Enum, str)) and u.u != "":
                raise ValueError(f"{type(v).__name__} value {v} with unit '{u.u}' is not allowed.")
            elif isinstance(v, int):
                if u.u != "":  # must be a 'hidden float'
                    types[-1] = float
        if len(types) == 1 or all(types[0] is t for t in types[1:]):  # all element types equal
            if issubclass(types[0], float):  # e.g. numpy.float64 is tracked as float
                return float
            else:
                return types[0]
        if any(t is float for t in types) and all(t is float or t is int for t in types):  # int&float -> float
            return float
        else:
            _units = tuple([u.u for u in units])
            raise ValueError(f"Auto-type cannot be determined for values {vals} with units {_units}")

    @classmethod
    def _auto_extreme(cls, var: PyType) -> tuple[float | bool, ...]:
        """Return the extreme values of the variable.

        Args:
            var: the variable for which to determine the extremes, represented by an instantiated object (example)

        Returns
        -------
            A tuple containing the minimum and maximum value the given variable can have
        """
        if isinstance(var, bool):
            return (False, True)
        elif isinstance(var, float):
            return (float("-inf"), float("inf"))
        elif isinstance(var, int):
            logger.critical(f"Range must be specified for int variable {cls} or use float.")
            return (var, var)  # restrict to start value
        elif isinstance(var, Enum):
            return (min(x.value for x in type(var)), max(x.value for x in type(var)))
        else:
            return tuple()  # return an empty tuple (no range specified, e.g. for str)

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
        _type = {"int": "Integer", "bool": "Boolean", "float": "Real", "str": "String", "Enum": "Enumeration"}[
            self.typ.__qualname__
        ]  # translation of python to FMI primitives. Same for all components
        do_use_start = use_start(causality=self._causality, variability=self._variability, initial=self._initial)
        svars = []
        a_der = self.primitive()  # d a_der /dt = self, or None
        for i in range(self._len):
            if self._len > 1:
                varname = ParsedVariable(self.name).as_string(index=str(i))
            else:
                varname = ParsedVariable(self.name).as_string(include=("parent", "var", "der"))
            sv = ET.Element(
                "ScalarVariable",
                {
                    "name": varname,
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
            #             if self._unit[ is None or (self._len>1 and self._unit[i].du is None):
            #                 "display" = (self.unit, 1.0)

            # detailed variable definition
            info = ET.Element(_type)
            if do_use_start:  # a start value is to be used
                info.attrib.update({"start": self.fmi_type_str(self._unit[i].from_base(self._start[i]))})
            if _type in ("Real", "Integer", "Enumeration"):  # range to be specified
                xmin = self.range[i].rng[0]
                if _type == "Real" and isinstance(xmin, float) and xmin == float("-inf"):
                    info.attrib.update({"unbounded": "true"})
                else:
                    info.attrib.update({"min": str(self._unit[i].from_base(xmin))})
                xmax = self.range[i].rng[1]
                if _type == "Real" and isinstance(xmax, float) and xmax == float("inf"):
                    info.attrib.update({"unbounded": "true"})
                else:
                    info.attrib.update({"max": str(self._unit[i].from_base(xmax))})
            if _type == "Real":  # other attributes apply only to Real variables
                info.attrib.update({"unit": self.unit[i].u})
                if isinstance(self._unit[i].du, str) and self.unit[i].du != self._unit[i].u:
                    info.attrib.update({"displayUnit": self._unit[i].du})  # type: ignore ## it is a str!
                if a_der is not None:
                    info.attrib.update({"derivative": str(a_der.value_reference + i + 1)})

            sv.append(info)
            svars.append(sv)
        return svars

    def primitive(self) -> Variable | None:
        """Determine the variable which self is the derivative of.
        Return None if self is not a derivative.
        """
        parsed = ParsedVariable(self.name)
        if parsed.der == 0:
            return None
        else:
            name = parsed.as_string(("parent", "var", "der"), simplified=True, primitive=True)
            return self.model.variable_by_name(name)
