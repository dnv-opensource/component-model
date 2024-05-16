import datetime
import tempfile
import uuid
import xml.etree.ElementTree as ET  # noqa: N817
from enum import Enum
from math import log
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from pint import UnitRegistry
from pythonfmu import DefaultExperiment, Fmi2Slave, FmuBuilder  # type: ignore
from pythonfmu import __version__ as pythonfmu_version
from pythonfmu.enums import Fmi2Causality as Causality  # type: ignore
from pythonfmu.enums import Fmi2Initial as Initial  # type: ignore
from pythonfmu.enums import Fmi2Variability as Variability  # type: ignore
from pythonfmu.fmi2slave import FMI2_MODEL_OPTIONS  # type: ignore

from .logger import get_module_logger
from .utils import read_model_description, xml_to_python_val
from .variable import Variable, VariableNP, variables_from_fmu

logger = get_module_logger(__name__, level=0)
Value: TypeAlias = str | int | float | bool | Enum


class ModelInitError(Exception):
    """Special error indicating that something is wrong with the boom definition."""

    pass


class ModelOperationError(Exception):
    """Special error indicating that something went wrong during crane operation (rotations, translations,calculation of CoM,...)."""

    pass


class ModelAnimationError(Exception):
    """Special error indicating that something went wrong during crane animation."""

    pass


class Model(Fmi2Slave):
    """Defines a model including some common model concepts, like variables and units.
    The model interface and the inner working of the model is missing here and must be defined in a given application.

    For a fully defined model instance shall:

    * define a full set of interface variables
    * set the current variable values
    * run the model in isolation for a time interval
    * retrieve updated variable values

    The following FMI concepts are (so far) not implemented:

    * TypeDefinitions. Instead of defining SimpleType variables, ScalarVariable variables are always based on the pre-defined types and details provided there
    * DisplayUnit. Variable units contain a Unit (the unit as used for inputs and outputs) and BaseUnit (the unit as used in internal model calculations, i.e. based on SI units).
      Additional DisplayUnit(s) are so far not defined/used. Unit is used for that purpose.
    * Explicit <Derivatives> (see <ModelStructure>) are so far not implemented. This could be done as an additional (optional) property in Variable.
      Need to check how this could be done with VariableNP
    * <InitialUnknowns> (see <ModelStructure>) are so far not implemented.
      Does overriding of setup_experiment(), enter_initialization_mode(), exit_initialization_mode() provide the necessary functionality?

    Args:
        name (str): name of the model, a valid modelDescription.xml or a fmu file
        author (str) = 'anonymous': The author of the model
        version (str) = '0.1': The version number of the model
        unit_system (str)='SI': The unit system to be used. self.ureg.default_system contains this information for all variables
        license (str)=None: License text or file name (relative to source files).
          If None, the BSD-3-Clause license as also used in the component_model package is used, with a modified copyright line
        copyright (str)=None: Copyright line for use in full license text. If None, an automatic copyright line is constructed from the author name and the file date.
        default_experiment (dict) = None: key/value dictionary for the default experiment setup (start_time,stop_time,step_size,tolerance)
        guid (str)=None: Unique identifier of the model (supplied or automatically generated)
        flags (dict)=None: Any of the defined FMI flags with a non-default value (see FMI 2.0.4, Section 4.3.1)
    """

    instances: list[str] = []

    def __init__(
        self,
        name,
        description: str = "A component model",
        author: str = "anonymous",
        version: str = "0.1",
        unit_system="SI",
        license: str | None = None,
        copyright: str | None = None,
        default_experiment: dict | None = None,
        flags: dict | None = None,
        guid=None,
        **kwargs,
    ):
        kwargs.update(
            {
                "modelName": name,
                "description": description,
                "author": author,
                "version": version,
                "copyright": copyright,
                "license": license,
                "guid": guid,
                "default_experiment": (
                    DefaultExperiment(None, None, None, None)
                    if default_experiment is None
                    else DefaultExperiment(**default_experiment)
                ),
            }
        )
        self.name = name
        if "instance_name" not in kwargs:
            kwargs["instance_name"] = self.name  # make_instancename(__name__)
        # self.check_and_register_instance_name(kwargs["instance_name"])
        if "resources" not in kwargs:
            kwargs["resources"] = None
        super().__init__(**kwargs)  # in addition, OrderedDict vars is initialized
        # Additional variables which are hidden here: .vars,
        self.description = description
        self.author = author
        self.version = version
        if guid is not None:
            self.guid = guid
        # use a common UnitRegistry for all variables:
        self.ureg = UnitRegistry(system=unit_system, autoconvert_offset_to_baseunit=True)
        self.copyright, self.license = self.make_copyright_license(copyright, license)
        ##        self.default_experiment = (DefaultExperiment(None, None, None, None) if default_experiment is None else DefaultExperiment(**default_experiment))
        self.guid = guid if guid is not None else uuid.uuid4().hex
        #        print("FLAGS", flags)
        self._units: dict[str, list] = {}  # def units and displayUnits (unitName:conversionFactor). => UnitDefinitions
        self.flags = self.check_flags(flags)
        self._dirty: dict[Variable, Any] = {}  # dirty compound variables. Used (set) during do_step()
        self.currentTime = 0  # keeping track of time when dynamic calculations are performed
        self._events: list[tuple] = []  # optional list of events activated on time during a simulation
        # Events consist of tuples of (time, changedVariable)

    def setup_experiment(self, start: float):
        """Minimum version of setup_experiment, just setting the start_time. In derived models this may not be enough."""
        self.start_time = start

    ## Other functions which can e overridden are
    # def enter_initialization_mode(self):
    #    def exit_initialization_mode(self):
    #        super().exit_initialization_mode()
    #        self.dirty_do() # run on_set on all dirty variables

    # def terminate(self):

    def do_step(self, currentTime, step_size):
        """Do a simulation step of size 'step_size at time 'currentTime.
        Note: this is only the generic part of this function. Models should call this first through super().do_step and then do their own stuff.
        """
        while len(self._events):  # Check whether any event is pending and set the respective variables
            (t0, (var, val)) = self._events[-1]
            if t0 <= currentTime:
                var.value = val
                self._events.pop()
            else:
                break
        self.dirty_do()  # run on_set on all dirty variables

        for var in self.vars.values():
            if var is not None and var.on_step is not None:
                var.on_step(currentTime, step_size)
        return True

    def _ensure_unit_registered(self, candidate: Variable):
        """Ensure that the displayUnit of a variable is registered.
        To register the units of a compound variable, the whole variable is entered
        and a recursive call to the underlying displayUnits is made.
        """
        unit_display = []
        if isinstance(candidate, VariableNP):
            for i in range(len(candidate)):  # recursive call to the components
                if candidate.displayUnit is None:
                    unit_display.append((candidate.unit[i], None))
                else:
                    unit_display.append((candidate.unit[i], candidate.displayUnit[i]))
        elif isinstance(candidate, Variable):
            unit_display.append((candidate.unit, candidate.displayUnit))
        # here the actual work is done
        for u, du in unit_display:
            if u not in self._units:  # the main unit is not yet registered
                self._units[u] = []  # main unit has no factor
            if du is not None:  # displayUnits are defined
                if du not in self._units[u]:
                    self._units[u].append(du)

    def register_variable(self, var: Variable, value0: Value | np.ndarray):
        """Register the variable 'var' as model variable. Set the initial value and add the unit if not yet used.
        Perform some checks and register the value_reference). The following should be noted.

        #. Only the first element of compound variables includes the variable reference,
           while the following sub-elements contain None, so that a (ScalarVariable) index is reserved.
        #. The variable var.name and var._unit must be set before calling this function.
        #. The call to super()... sets the value_reference, getter and setter of the variable
        """
        for idx, v in self.vars.items():
            msg = f"Variable name {var.name} is not unique in model {self.name}. Already used as reference {idx}"
            assert v is None or v.name != var.name, msg
        setattr(self, var.name, value0)  # ensure that the model has the value as attribute
        #        super().register_variable(var)
        variable_reference = len(self.vars)
        self.vars[variable_reference] = var
        var.value_reference = variable_reference  # Set the unique value reference
        if var.getter is None:
            var.getter = lambda: getattr(self, var.local_name)
        if var.setter is None and hasattr(self, var.local_name) and var.variability != Variability.constant:
            if isinstance(var, VariableNP):
                var.setter = lambda v: setattr(self, var.local_name, np.array(v, dtype=var.type))
            else:
                var.setter = lambda v: setattr(self, var.local_name, v)

        logger.info(f"REGISTER Variable {var.name}. getter: {var.getter}, setter: {var.setter}")
        if isinstance(var, VariableNP):
            for i in range(1, len(var)):
                self.vars[var.value_reference + i] = None  # marking that this is a sub-element
        if value0 is not None:
            var.setter(value0, None)
        self._ensure_unit_registered(var)

    def ensure_dirty(self, var: Variable, value: Value | tuple[Value] | np.ndarray, idx: int | None = None):
        """Ensure that the variable var is registered in self._dirty
        and that the (new) value is listed there.
        Either single elements or a whole array can be set/overwritten.
        Scalar variable values are stored as list of a single value.
        """
        is_scalar = not isinstance(var, VariableNP)
        if var in self._dirty:
            if is_scalar:  # already registered scalar
                assert idx is None, f"The variable {var.name} has no indices. Found {idx}."
                self._dirty[var] = value
            else:
                if idx is None:  # already registered vector. All elements
                    msg = f"Value {value} should be vector of length {len(var)}."
                    assert isinstance(value, (tuple, np.ndarray)) and len(var) == len(value), msg
                    self._dirty[var] = value
                else:  # already registered vector. Single element
                    msg = f"Erroneous index {idx} or value {value} setting a vector element of {var.name}."
                    assert isinstance(value, (str, int, float, bool, Enum)) and 0 <= idx < len(var), msg
                    self._dirty[var][idx] = value
        else:
            if is_scalar:  # new dirty scalar
                assert idx is None, f"The new variable {var.name} has no indices. Found {idx}."
                self._dirty.update({var: value})
            else:
                if idx is None:  # new vector. All elements
                    msg = f"Value {value} should be vector of length {len(var)}."
                    assert isinstance(value, (tuple, np.ndarray)) and len(var) == len(value), msg
                    self._dirty.update({var: value})
                else:  # new vector. Single element
                    msg = f"Erroneous index {idx} or value {value} setting a vector element of {var.name}."
                    assert isinstance(value, (str, int, float, bool, Enum)) and 0 <= idx < len(var), msg
                    self._dirty.update({var: getattr(self, var.local_name)})  # start with current value
                    self._dirty[var][idx] = value

    def is_dirty(self, var: Variable):
        """Check whether the variable var is listed in self._dirty."""
        return var in self._dirty

    def get_from_dirty(self, var: Variable):
        """Get the 'staged' value from the dirty dict."""
        assert var in self._dirty, f"Variable {var.name} not found in _dirty, as ecpected"
        return self._dirty[var]

    def dirty_do(self):
        """Run on_set on all dirty variables."""
        for v, val in self._dirty.items():
            setattr(self, v.local_name, val if v.on_set is None else v.on_set(val))
        self._dirty = {}

    @property
    def units(self):
        return self._units

    def add_variable(self, *args, **kwargs):
        """Add a variable, automatically including the owner model in the instantiation call."""
        return Variable(self, *args, **kwargs)

    def add_event(self, time: float | None = None, event: tuple | None = None):
        """Register a new event to the event list. Ensure that the list is sorted.
        Note that the event mechanism is mainly used for model testing, since normally events are initiated by input variable changes.

        Args:
            time (float): the time at which the event shall be issued. If None, the event shall happen immediatelly
            event (tuple): tuple of the variable (by name or object) and its changed value
        """
        if event is None:
            return  # no action
        var = event[0] if isinstance(event[0], Variable) else self.variable_by_name(event[0])
        assert var is not None, "Trying to add event related to unknown variable " + str(event[0]) + ". Ignored."
        if time is None:
            self._events.append((-1, (var, event[1])))  # sorted wrt. decending time negative times denote 'immediate'
        else:
            if not len(self._events):
                self._events.append((time, (var, event[1])))
            else:
                for i, (t, _) in enumerate(self._events):
                    if t < time:
                        self._events.insert(i, (time, (var, event[1])))
                        break

    def variable_by_name(self, name: str, msg: str | None = None):
        """Return Variable object related to name, or None, if not found.
        For compound variables, the parent variable is returned irrespective of whether the '.#' is included or not
        If msg is not None, an error is raised and the message provided.
        """
        for var in self.vars.values():
            if var is not None and name.startswith(var.name):
                if len(name) == len(var.name):  # identical (single compound variable)
                    return var
                else:
                    try:
                        sub = int(name[len(var.name) + 1 :])
                        if sub < len(var):
                            return var
                    except Exception:
                        pass
        if msg is not None:
            raise ModelInitError(msg)
        return None

    def variable_by_value(self, value):
        """Get the variable object from the current value (which is owned by the model)."""
        for var in self.vars.values():
            if id(var.getter()) == id(value):
                return var
        return None

    def xml_unit_definitions(self):
        """Make the xml element for the unit definitions used in the model. See FMI 2.0.4 specification 2.2.2."""
        defs = ET.Element("UnitDefinitions")
        for u in self._units:
            ubase = self.ureg(u).to_base_units()
            dim = ubase.dimensionality
            exponents = {}
            for key, value in {
                "mass": "kg",
                "length": "m",
                "time": "s",
                "current": "A",
                "temperature": "K",
                "substance": "mol",
                "luminosity": "cd",
            }.items():
                if "[" + key + "]" in dim:
                    exponents.update({value: str(int(dim["[" + key + "]"]))})
            if (
                "radian" in str(ubase.units)
            ):  # radians are formally a dimensionless quantity. To include 'rad' as specified in FMI standard this dirty trick is used
                # udeg = str(ubase.units).replace("radian", "degree")
                # print("EXPONENT", ubase.units, udeg, log(ubase.magnitude), log(self.ureg('degree').to_base_units().magnitude))
                exponents.update(
                    {"rad": str(int(log(ubase.magnitude) / log(self.ureg("degree").to_base_units().magnitude)))}
                )

            unit = ET.Element("Unit", {"name": u})
            base = ET.Element("BaseUnit", exponents)
            base.attrib.update({"factor": str(self.ureg(u).to_base_units().magnitude)})
            unit.append(base)
            for dU in self._units[u]:  # list also the displayUnits (if defined)
                unit.append(ET.Element("DisplayUnit", {"name": dU[0], "factor": str(dU[1])}))
            defs.append(unit)
        return defs

    def make_instancename(self, base):
        """Make a new (unique) instance name, using 'base_#."""
        ext = []
        for name in Model.instances:
            if name.startswith(base + "_") and name[len(base) + 1 :].isnumeric():
                ext.append(int(name[len(base) + 1 :]))
        return base + "_" + "0" if not len(ext) else str(sorted(ext)[-1] + 1)

    def check_and_register_instance_name(self, iName):
        assert all(name != iName for name in Model.instances), f"The instance name {iName} is not unique"
        Model.instances.append(iName)

    def make_copyright_license(self, copyright: str | None = None, license: str | None = None):
        """Prepare a copyright notice (one line) and a license text (without copyright line).
        If license is None, the license text of the component_model package is used (BSD-3-Clause).
        If copyright is None, a copyright text is construced from self.author and the file date.
        """
        import datetime
        import os

        if license is None:
            license = """Permission is hereby granted, free of charge, to any person obtaining a copy
            of this software and associated documentation files (the "Software"), to deal
            in the Software without restriction, including without limitation the rights
            to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
            copies of the Software, and to permit persons to whom the Software is
            furnished to do so, subject to the following conditions:

            The above copyright notice and this permission notice shall be included in all
            copies or substantial portions of the Software.

            THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
            IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
            FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
            AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
            LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
            OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
            SOFTWARE."""
        license = "".join(line.strip() + "\n" for line in license.split("\n"))  # remove whitespace in lines
        if license.partition("\n")[0].lower().startswith("copyright"):
            copyright1 = license.partition("\n")[0].strip()
            license = license.partition("\n")[2].strip()
        else:
            copyright1 = None

        if copyright is None:
            if copyright1 is None:  # make a new one
                copyright = (
                    "Copyright (c) "
                    + str(datetime.datetime.fromtimestamp(os.path.getatime(__file__)).year)
                    + " "
                    + self.author
                )
            else:
                copyright = copyright1

        return (copyright, license)

    # =====================
    # FMU-related functions
    # =====================
    @staticmethod
    def build(
        scriptFile: str | None = None,
        project_files: list | None = None,
        dest: str = ".",
        documentation_folder: Path | None = None,
    ):
        """!!Note: Since the build process is linked to the scriptFile and not to the class,
        it is currently not possible to define several FMUs in one script. Max 1 per file.
        """
        if scriptFile is None:
            scriptFile = __file__
        if project_files is None:
            project_files = []
        project_files.append(Path(__file__).parents[0])
        with tempfile.TemporaryDirectory() as documentation_dir:
            doc_dir = Path(documentation_dir)
            license_file = doc_dir / "licenses" / "license.txt"
            license_file.parent.mkdir()
            license_file.write_text("Dummy license")
            index_file = doc_dir / "index.html"
            index_file.write_text("dummy index")
            asBuilt = FmuBuilder.build_FMU(
                scriptFile,
                project_files=project_files,
                dest=dest,
                documentation_folder=doc_dir,
            )  # , xFunc=None)
            return asBuilt

    def to_xml(self, model_options: dict | None = None) -> ET.Element:
        """Build the XML FMI2 modelDescription.xml tree. (adapted from Fmi2Slave.to_xml()).

        Args:
            model_options ({str, str}) : FMU model options

        Returns
        -------
            (xml.etree.TreeElement.Element) XML description of the FMU
        """
        if model_options is None:
            model_options = {}
        t = datetime.datetime.now(datetime.timezone.utc)
        date_str = t.isoformat(timespec="seconds")

        attrib = dict(
            fmiVersion="2.0",
            modelName=self.modelName,
            guid=f"{self.guid!s}",
            generationTool=f"PythonFMU {pythonfmu_version}",
            generationDateAndTime=date_str,
            variableNamingConvention="structured",
        )
        if self.description is not None:
            attrib["description"] = self.description
        if self.author is not None:
            attrib["author"] = self.author
        if self.license is not None:
            attrib["license"] = self.license
        if self.version is not None:
            attrib["version"] = self.version
        if self.copyright is not None:
            attrib["copyright"] = self.copyright

        root = ET.Element("fmiModelDescription", attrib)

        options = dict()
        for option in FMI2_MODEL_OPTIONS:
            value = model_options.get(option.name, option.value)
            options[option.name] = str(value).lower()
        options["modelIdentifier"] = self.modelName
        options["canNotUseMemoryManagementFunctions"] = "true"

        ET.SubElement(root, "CoSimulation", attrib=options)

        root.append(self.xml_unit_definitions())

        if len(self.log_categories) > 0:
            categories = ET.SubElement(root, "LogCategories")
            for category, description in self.log_categories.items():
                categories.append(
                    ET.Element(
                        "Category",
                        attrib={"name": category, "description": description},
                    )
                )
        if self.default_experiment is not None:
            attrib = dict()
            for a, e in [
                ("start_time", "startTime"),
                ("stop_time", "stopTime"),
                ("step_size", "step_size"),
                ("tolerance", "tolerance"),
            ]:
                if getattr(self.default_experiment, a, None) is not None:
                    attrib[e] = str(getattr(self.default_experiment, a))
            ET.SubElement(root, "DefaultExperiment", attrib)

        variables = self._xml_modelvariables()
        root.append(variables)  # append <ModelVariables>

        structure = ET.SubElement(root, "ModelStructure")
        structure.append(self._xml_structure_outputs())
        initialunknowns = self._xml_structure_initialunknowns()
        if len(initialunknowns):
            structure.append(initialunknowns)
        return root

    def _xml_modelvariables(self):
        """Generate the FMI2 modelDescription.xml sub-tree <ModelVariables>."""
        mv = ET.Element("ModelVariables")
        for var in self.vars_iter():
            var.xml_scalarvariables(mv)
        return mv

    def _xml_structure_outputs(self):
        """Generate the FMI2 modelDescription.xml sub-tree <ModelStructure><Outputs>.
        Exactly all variables with causality='output' must be in this list.
        ToDo: implement support for variable dependencies and add as attribute here.
        """
        out = ET.Element("Outputs")

        for v in filter(lambda v: v is not None and v.causality == Causality.output, self.vars.values()):
            if len(v) == 1:
                out.append(ET.Element("Unknown", {"index": str(v.value_reference + 1)}))
            else:
                for i in range(len(v)):
                    out.append(ET.Element("Unknown", {"index": str(v.value_reference + i + 1)}))
        return out

    def _xml_structure_initialunknowns(self):
        """Generate the FMI2 modelDescription.xml sub-tree <ModelStructure><InitialUnknowns>.
        Ordered list of all exposed Unknowns in Initialization mode. All variables with (see page 60 FMI2 spec).

        * causality = 'output' and initial = 'approx' or 'calculated'
        * causality = 'calculatedParameter'
        * all continuous-time states and all state derivatives with initial = 'approx' or 'calculated'

        ToDo: implement support for states and derivatives and add as attribute here.
        ToDo: implement support for variable dependencies and add as attribute here.
        """
        init = ET.Element("InitialUnknowns")

        for v in filter(
            lambda v: v is not None
            and (
                (v.causality == Causality.output and v.initial in (Initial.approx, Initial.calculated))
                or (v.causality == Causality.calculatedParameter)
            ),
            self.vars.values(),
        ):
            if len(v) == 1:
                init.append(ET.Element("Unknown", {"index": str(v.value_reference + 1)}))
            else:
                for i in range(len(v)):
                    init.append(ET.Element("Unknown", {"index": str(v.value_reference + i + 1)}))
        return init

    @staticmethod
    def check_flags(flags: dict | None):
        """Check and collect provided flags dictionary and return the non-default flags.
        Any of the defined FMI flags with a non-default value (see FMI 2.0.4, Section 4.3.1).
        .. todo:: Check also whether the model actually provides these features.
        """
        _flags = {}
        if flags is not None:
            for flag, default in {
                "needsExecutionTool": False,
                "canHandleVariableCommunicationStepSize": False,
                "canInterpolateInputs": False,
                "maxOutputDerivativeOrder": 0,
                "canRunAsynchchronously": False,
                "canBeInstantiatedOnlyOncePerProcess": False,
                "canNotUseMemoryManagementFunctions": False,
                "canGetAndSetFMUstate": False,
                "canSerializeFMUstate": False,
                "providesDirectionalDerivative": False,
            }.items():
                if flag in flags and flags[flag] != default and isinstance(flags[flag], type(default)):
                    _flags.update({flag: flags[flag]})
        return _flags

    def vars_iter(self, key=None):
        """Iterate over model variables ('vars'). The returned variables depend on 'key' (see below).

        Args:
            key: filter for returned variables. The following possibilities exist:

            * None: All variables are returned
            * type: A type designator (int, float, bool, Enum, str), returning only variables matching on this type
            * causality: A Causality value, returning only one causality type, e.g. Causality.input
            * variability: A Variability value, returning only one variability type, e.g. Variability.fixed
            * callable: Any bool function of model variable object

        If typ and causality are None, all variables are included, otherwise only the specified type/causality.
        The returned list can be indexed to retrieve given valueReference variables.
        """
        if key is None:  # all variables
            for v in self.vars.values():
                if v is not None:
                    yield v
        elif isinstance(key, type):  # variable type iterator
            for v in self.vars.values():
                if v is not None and v.type == key:
                    yield v
        elif isinstance(key, Causality):
            for v in self.vars.values():
                if v is not None and v.causality == key:
                    yield v
        elif isinstance(key, Variability):
            for v in self.vars.values():
                if v is not None and v.variability == key:
                    yield v
        elif callable(key):
            for v in self.vars.values():
                if v is not None and key(v):
                    yield v

        else:
            raise KeyError(f"Unknown iteration key {key} in 'vars_iter'")

    # ================
    # Need to over-write the get_ and set_ variable access functions, since we need to deal with compound variables
    def ref_to_var(self, vr: int):
        """Find Variable and sub-index (for compound variable), based on a valueReference value."""
        _vr = vr
        while True:
            var = self.vars[_vr]
            if var is None:
                _vr -= 1
            else:  # found the base of the variable
                return (var, vr - _vr)

    def _var_iter(self, vrs: list[int]):
        """Convert a list of value_reference integers into a variable object iterator.
        Take also compound variables into account:
          If all variables are listed, include the compound object in the result.
          Otherwise include the compound object and an index.
        """
        it = enumerate(vrs.__iter__())  # get an enumerated iterator over vrs
        for i, vr in it:
            sub = None
            assert vr < len(self.vars), f"Variable with valueReference={vr} does not exist in model {self.name}"
            var = self.vars[vr]
            if var is None:  # isolated element(s) of compound variable
                var, sub = self.ref_to_var(vr)
            elif isinstance(var, VariableNP):
                sub = 0
            # At this point we should have a variable object
            if (
                isinstance(var, VariableNP) and i + len(var) <= len(vrs) and vr + len(var) - 1 == vrs[i + len(var) - 1]
            ):  # compound variable and all elements included
                for _ in range(len(var) - 1):  # spool to the last element
                    i, vr = next(it)
                sub = None
            #                print(f"VAR_ITER whole {var.name} at {vr}")
            #            print(f"_VAR_ITER {var.name}[{sub}], i:{i}, vr:{vr}")
            yield (var, sub)

    def _get(self, vrs: list, typ: type) -> list:
        """Get variables of all types based on references.
        This method is called by get_xxx and translates to fmi2GetXxx.
        """
        values = list()
        for var, sub in self._var_iter(vrs):
            assert var.type == typ, f"Invalid type in 'get_{typ}'. Found variable {var.name} with type {var.type}"
            val = var.getter()
            if isinstance(var, VariableNP):
                if sub is None:
                    values.extend(val)
                else:
                    values.append(val[sub])
            else:
                values.append(val)
        #            print(f"_GET {var.name}[{sub}], type:{type(var)}, values:{values}")
        return values

    def get_integer(self, vrs):
        return self._get(vrs, int)

    def get_real(self, vrs):
        return self._get(vrs, float)

    def get_boolean(self, vrs):
        return self._get(vrs, bool)

    def get_string(self, vrs):
        return self._get(vrs, str)

    def _set(self, vrs: list, values: list, typ: type):
        """Set variables of all types. This method is called by set_xxx and translates to fmi2SetXxx.
        Variable range check, unit check and type check are performed by setter() function.
        on_set (if defined) is only run if the whole variable (all elements) are set.
        """
        idx = 0
        for var, sub in self._var_iter(vrs):
            assert var.type == typ, f"Invalid type in 'get_{typ}'. Found variable {var.name} with type {var.type}"
            if isinstance(var, VariableNP):
                if sub is None:  # set the whole vector
                    var.setter(values[idx : idx + len(var)], idx=None)
                    idx += len(var) - 1
                else:
                    var.setter(values[idx], sub)
            else:  # simple Variable
                var.setter(values[idx], idx=None)
            idx += 1

    def set_integer(self, vrs: list, values: list):
        self._set(vrs, values, int)

    def set_real(self, vrs: list, values: list):
        self._set(vrs, values, float)

    def set_boolean(self, vrs: list, values: list):
        self._set(vrs, values, bool)

    def set_string(self, vrs: list, values: list):
        self._set(vrs, values, str)

    def _get_fmu_state(self) -> dict:
        """Get the value of all referenced variables of the model.
        Note that also compound variables are saved in a single slot.
        """
        state = dict()
        for var in self.vars.values():
            if var is not None:
                state[var.local_name] = getattr(self, var.local_name)
        return state

    def _set_fmu_state(self, state: dict):
        """Set all variables as saved in state.
        Note: Compound variables are expected in a single slot.
        """
        for name, value in state.items():
            #            if var is None:  # not previously registered (seems to be allowed!?)
            setattr(self, name, value)


#             elif var.on_set is None or var.causality == Causality.output:
#                 var.value = value
#             else:
#                 var.value = var.on_set(value)


# ==========================================
# Open Simulation Platform related functions
# ==========================================
def make_osp_system_structure(
    name: str = "OspSystemStructure",
    models: dict | None = None,
    connections: tuple = (),
    version: str = "0.1",
    start: float = 0.0,
    base_step: float = 0.01,
    algorithm: str = "fixedStep",
):
    """Prepare a OspSystemStructure xml file according to `OSP configuration specification <https://open-simulation-platform.github.io/libcosim/configuration>`_.

    Args:
        name (str)='OspSystemStructure': the name of the system model, used also as file name
        models (dict)={}: dict of models (in OSP called 'simulators'). A model is represented by a dict element modelName : {property:prop, variable:value, ...}
        connections (tuple)=(): tuple of model connections. Each connection is defined through a tuple of (model, variable, model, variable), where variable can be a tuple defining a variable group
        version (str)='0.1': The version of the system model
        start (float)=0.0: The simulation start time
        base_step (float)=0.01: The base stepSize of the simulation. The exact usage depends on the algorithm chosen
        algorithm (str)='fixedStep': The name of the algorithm

        ??ToDo: better stepSize control in dependence on algorithm selected, e.g. with fixedStep we should probably set all step sizes to the minimum of everything?
    """

    def make_simulators():
        """Make the <simulators> element (list of component models)."""

        def make_initial_value(var: str, val: bool | int | float | str):
            """Make a <InitialValue> element from the provided var dict."""
            _type = {bool: "Boolean", int: "Integer", float: "Real", str: "String"}[type(val)]
            initial = ET.Element("InitialValue", {"variable": var})
            ET.SubElement(
                initial,
                _type,
                {"value": ("true" if val else "false") if isinstance(val, bool) else str(val)},
            )
            return initial

        simulators = ET.Element("Simulators")
        if len(models):
            for m, props in models.items():
                # Note: instantiated model names might be small, but FMUs are based on class names and are therefore capitalized
                simulator = ET.Element(
                    "Simulator",
                    {
                        "name": m,
                        "source": props.get("source", m[0].upper() + m[1:] + ".fmu"),
                        "stepSize": str(props.get("stepSize", base_step)),
                    },
                )
                initial = ET.SubElement(simulator, "InitialValues")
                for prop, value in props.items():
                    if prop not in ("source", "stepSize"):
                        initial.append(make_initial_value(prop, value))
                simulators.append(simulator)
            #            print(f"Model {m}: {simulator}. Length {len(simulators)}")
            #            ET.ElementTree(simulators).write("Test.xml")
            return simulators

    def make_connections():
        """Make the <connections> element from the provided con."""
        cons = ET.Element("Connections")
        m1, v1, m2, v2 = connections
        if isinstance(v1, (tuple, list)):  # group connection (e.g. a VariableNP)
            if not isinstance(v2, (tuple, list)) or len(v2) != len(v1):
                raise ModelInitError(
                    f"Something wrong with the vector connection between {m1} and {m2}. Variable vectors do not match."
                )
            for i in range(len(v1)):
                con = ET.Element("VariableConnection")
                ET.SubElement(con, "Variable", {"simulator": m1, "name": v1[i]})
                ET.SubElement(con, "Variable", {"simulator": m2, "name": v2[i]})
                cons.append(con)
        else:  # single connection
            con = ET.Element("VariableConnection")
            ET.SubElement(con, "Variable", {"simulator": m1, "name": v1})
            ET.SubElement(con, "Variable", {"simulator": m2, "name": v2})
            cons.append(con)
        return cons

    osp = ET.Element(
        "OspSystemStructure",
        {
            "xmlns": "http://opensimulationplatform.com/MSMI/OSPSystemStructure",
            "version": version,
        },
    )
    osp.append(make_simulators())
    osp.append(make_connections())
    tree = ET.ElementTree(osp)
    ET.indent(tree, space="   ", level=0)
    tree.write(name + ".xml", encoding="utf-8")


def model_from_fmu(fmu: str | Path, provideMsg: bool = False, sep="."):
    """Generate a ComponentModel from an FMU (excluding the inner working functions like 'do_step'.
    Still this is useful for convenient access to model information like variables.
    Note: structured variables with name: <name>.i, with otherwise equal causality, variability, initial
    and consecutive index and valueReference are stored as VariableNP.
    .. ToDo:: <UnitDefinitions>, <LogCategories>.

    Args:
        fmu (str, Path): the FMU file which is to be read. can be the full FMU zipfile, the modelDescription.xml or a equivalent string
        provideMsg (bool): Optional possibility to provide messages during the process (for debugging purposes)
        sep (str)='.': separation used for structured variables (both for sub-systems and variable names)

    Returns
    -------
        Model object
    """

    el = read_model_description(fmu)
    defaultexperiment = el.find(".//DefaultExperiment")
    de = {} if defaultexperiment is None else defaultexperiment.attrib
    co_flags = el.find(".//CoSimulation")
    flags = {} if co_flags is None else {key: xml_to_python_val(val) for key, val in co_flags.attrib.items()}
    model = Model(
        name=el.attrib["modelName"],
        description=el.get("description", f"Component model object generated from {fmu}"),
        author=el.get("author", "anonymous"),
        version=el.get("version", "0.1"),
        unit_system="SI",
        license=el.get("license", None),
        copyright=el.get("copyright", None),
        guid=el.get("guid", None),
        default_experiment={
            "start_time": float(de.get("start", 0.0)),
            "stop_time": float(de["stopTime"]) if "stopTime" in de else None,
            "step_size": float(de["stepSize"]) if "stepSize" in de else None,
            "tolerance": float(de["tolerance"]) if "tolerance" in de else None,
        },
        flags=flags,
    )
    variables_from_fmu(model, el.find(".//ModelVariables"), sep=sep)
    return model
