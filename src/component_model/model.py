"""Python module as container for a (FMU) component model."""

import datetime
import os
import tempfile
import uuid
import xml.etree.ElementTree as ET
from abc import abstractmethod
from enum import Enum
from math import log
from pathlib import Path
from typing import TypeAlias

import numpy as np
from pint import UnitRegistry
from pythonfmu import Fmi2Slave, FmuBuilder  # type: ignore
from pythonfmu import __version__ as pythonfmu_version
from pythonfmu.default_experiment import DefaultExperiment
from pythonfmu.enums import Fmi2Causality as Causality  # type: ignore
from pythonfmu.enums import Fmi2Variability as Variability  # type: ignore
from pythonfmu.fmi2slave import FMI2_MODEL_OPTIONS  # type: ignore

from component_model.caus_var_ini import Initial
from component_model.utils.logger import get_module_logger
from component_model.variable import Variable

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
    """Defines a model complying to the `FMI standard <https://fmi-standard.org/>`_,
    including some common model concepts, like variables and units.
    The package extends the `PythonFMU package <https://github.com/NTNU-IHB/PythonFMU>`_.

    The model interface and the inner working of the model is missing here
    and must be defined in a given application, extending `Model`.

    A fully defined model shall at least:

    * Define a full set of interface variables, including default start values, setter and getter functions.
      See `variable` module.
    * Extend the `do_step(time, dt)` member function, running the application model in isolation for a time interval.
      Make sure that `super().do_step(time, dt)` is always called first in the extended function.
    * Optionally extend any other fmi2 function, i.e.

       - def setup_experiment(self, start)
       - def enter_initialization_mode(self):
       - def exit_initialization_mode(self):
       - def terminate(self):


    The following FMI concepts are (so far) not implemented:

    * TypeDefinitions. Instead of defining SimpleType variables,
      ScalarVariable variables are always based on the pre-defined types and details provided there.
    * Explicit <Derivatives> (see <ModelStructure>) are so far not implemented.
      This could be done as an additional (optional) property in Variable.

    Args:
        name (str): name of the model. The name is also used to construct the FMU file name.
        author (str) = 'anonymous': The author of the model
        version (str) = '0.1': The version number of the model
        unit_system (str)='SI': The unit system to be used.
          `self.ureg.default_system` contains this information for all variables
        license (str)=None: License text or file name (relative to source files).
          If None, the BSD-3-Clause license as also used in the component_model package is used, with a modified copyright line
        copyright (str)=None: Copyright line for use in full license text. If None, an automatic copyright line is constructed from the author name and the file date.
        default_experiment (dict) = None: key/value dictionary for the default experiment setup.
          Valid keys: startTime,stopTime,stepSize,tolerance
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
        default_experiment: dict[str, float] | None = None,
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
            }
        )
        self.name = name
        if "instance_name" not in kwargs:  # NOTE: within builder.py this is always changed to 'dummyInstance'
            kwargs["instance_name"] = self.name  # make_instancename(__name__)
        if "resources" not in kwargs:
            kwargs["resources"] = None
        super().__init__(**kwargs)  # in addition, OrderedDict vars is initialized
        # Additional variables which are hidden here: .vars,
        # PythonFMU sets the default_experiment and the following items always to None! Correct it here
        if default_experiment is None:
            self.default_experiment = DefaultExperiment(start_time=0.0, stop_time=1.0, step_size=0.01, tolerance=1e-3)
        else:
            self.default_experiment = DefaultExperiment(
                start_time=default_experiment.get("startTime", 0.0),
                stop_time=default_experiment.get("stopTime", 1.0),
                step_size=default_experiment.get("stepSize", 0.01),
                tolerance=default_experiment.get("tolerance", 1e-3),
            )
        self.description = description
        self.author = author
        self.version = version
        if guid is not None:
            self.guid = guid
        # use a common UnitRegistry for all variables:
        self.ureg = UnitRegistry(system=unit_system)
        self.copyright, self.license = self.make_copyright_license(copyright, license)
        self.guid = guid if guid is not None else uuid.uuid4().hex
        #        print("FLAGS", flags)
        self._units: dict[str, list] = {}  # def units and display units (unitName:conversionFactor). => UnitDefinitions
        self.flags = self.check_flags(flags)
        self._dirty: list = []  # dirty compound variables. Used by (set) during do_step()
        self.currentTime = 0.0  # keeping track of time when dynamic calculations are performed

    def setup_experiment(self, start: int | float = 0.0):
        """Minimum version of setup_experiment, just setting the start_time. Derived models may need to extend this."""
        self.start_time = start

    def exit_initialization_mode(self):
        """Initialize the model after initial variables are set."""
        super().exit_initialization_mode()
        self.dirty_do()  # run on_set on all dirty variables

    @abstractmethod  # mark the class as 'still abstract'
    def do_step(self, time: int | float, dt: int | float):
        """Do a simulation step of size 'step_size at time 'currentTime.
        Note: this is only the generic part of this function. Models should call this first through super().do_step and then do their own stuff.
        """
        self.currentTime = time
        self.dirty_do()  # run on_set on all dirty variables

        for var in self.vars.values():
            if var is not None and var.on_step is not None:
                var.on_step(time, dt)
        return True

    def _unit_ensure_registered(self, candidate: Variable):
        """Ensure that the display of a variable is registered.
        To register the units of a compound variable, the whole variable is entered
        and a recursive call to the underlying display units is made.
        """
        unit_display = []
        for i in range(len(candidate)):
            if candidate.display[i] is None:
                unit_display.append((candidate.unit[i], None))
            else:
                unit_display.append((candidate.unit[i], candidate.display[i]))
        # here the actual work is done
        for u, du in unit_display:
            if u not in self._units:  # the main unit is not yet registered
                self._units[u] = []  # main unit has no factor
            if du is not None:  # displays are defined
                if not len(self._units[u]) or all(
                    du[0] not in self._units[u][i][0] for i in range(len(self._units[u]))
                ):
                    self._units[u].append(du)

    def register_variable(
        self,
        var: Variable,
        start: tuple | list | np.ndarray,
        value_reference: int | None = None,
    ):
        """Register the variable 'var' as model variable. Set the initial value and add the unit if not yet used.
        Perform some checks and register the value_reference. The following should be noted.

        #. Only the first element of compound variables includes the variable reference,
           while the following sub-elements contain None, so that a (ScalarVariable) index is reserved.
        #. The variable var.name and var.unit must be set before calling this function.
        #. The call to super()... sets the value_reference, getter and setter of the variable
        """
        for idx, v in self.vars.items():
            if v is not None and v.name == var.name:
                raise KeyError(f"Variable {var.name} already used as index {idx} in model {self.name}") from None
        # ensure that the model has the value as attribute:
        setattr(self, var.name, np.array(start, var.typ) if len(var) > 1 else start[0])
        if value_reference is None:  # automatic valueReference
            vref = len(self.vars)
        else:
            vref = value_reference
        self.vars[vref] = var
        var.value_reference = vref  # Set the unique value reference
        # logger.info(f"REGISTER Variable {var.name}. getter: {var.getter}, setter: {var.setter}")
        for i in range(1, len(var)):
            self.vars[var.value_reference + i] = None  # marking that this is a sub-element
        self._unit_ensure_registered(var)

    def dirty_ensure(self, var: Variable):
        """Ensure that the variable var is registered in self._dirty.

        The `dirty` mechanism is used when elements of compound variables are changed
        and a `on_set` function is defined, such that on_set() is run exactly once and when all elements are changed.
        """
        if var not in self._dirty:
            self._dirty.append(var)

    @property
    def dirty(self):
        return self._dirty

    def dirty_do(self):
        """Run on_set on all dirty variables."""
        for var in self._dirty:
            if var.on_set is not None:
                val = var.on_set(getattr(var.owner, var.local_name))
                setattr(var.owner, var.local_name, val)
        self._dirty = []

    @property
    def units(self):
        return self._units

    def add_variable(self, *args, **kwargs):
        """Add a variable, automatically including the owner model in the instantiation call.
        The function represents an alternative method for defining interface variables
        automatically adding the mandatory first `model` argument.
        """
        return Variable(self, *args, **kwargs)

    def variable_by_name(self, name: str) -> Variable:
        """Return Variable object related to name, or None, if not found.
        For compound variables, the parent variable is returned
        irrespective of whether an index (`[#]`) is included or not
        If msg is not None, an error is raised and the message provided.
        """
        for var in self.vars.values():
            if var is not None and name.startswith(var.name):
                if len(name) == len(var.name):  # identical (single compound variable)
                    return var
                else:
                    ext = name[len(var.name) :]
                    if ext[0] == "[" and ext[-1] == "]":
                        try:
                            sub = int(ext[1:-1])
                            if 0 <= sub < len(var):
                                return var
                        except Exception:
                            pass
        raise KeyError(f"Variable {name} not found in model {self.name}") from None

    def make_copyright_license(self, copyright: str | None = None, license: str | None = None):
        """Prepare a copyright notice (one line) and a license text (without copyright line).
        If license is None, the license text of the component_model package is used (BSD-3-Clause).
        If copyright is None, a copyright text is construced from self.author and the file date.
        """
        import datetime

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
                copyright = "Copyright (c) " + str(datetime.datetime.now().year) + " " + self.author
            else:
                copyright = copyright1

        return (copyright, license)

    @staticmethod
    def ensure_requirements(existing_file: str | Path | None, temp_file: Path) -> Path:
        """Return the path to the component-model requirements file."""
        if existing_file is None:
            requirements = ["numpy", "pint"]
            temp_file.write_text("\n".join(requirements))
            return temp_file
        with open(existing_file) as file:
            requirements = file.read().splitlines()

        if "numpy" not in requirements:
            requirements.append("numpy")
        if "pint" not in requirements:
            requirements.append("pint")

        temp_file.write_text("\n".join(requirements))
        return temp_file

    # =====================
    # FMU-related functions
    # =====================
    @staticmethod
    def build(
        script: str | Path = "",
        project_files: list[str | Path] | None = None,
        dest: str | os.PathLike[str] = ".",
        documentation_folder: Path | None = None,
    ):
        """Build the FMU, resulting in the model-name.fmu file.

        !!Note: Since the build process is linked to the script and not to the class,
        it is currently not possible to define several FMUs in one script. Max 1 per file.

        Args:
           script (str) = "": The scriptfile (xxx.py) in which the model class is defined. This file if ""
           project_files (list): Optional list of additional files to include in the build (relative to script)
           dest (str) = '.': Optional destination folder for the FMU.
           documentation_folder (Path): Optional folder with additional model documentation files.
        """
        if script is None:
            script = __file__
        if project_files is None:
            project_files = []
        project_files.append(Path(__file__).parents[0])

        # Make sure the dest path is of type Patch
        dest = dest if isinstance(dest, Path) else Path(dest)

        with (
            tempfile.TemporaryDirectory() as documentation_dir,
            tempfile.TemporaryDirectory() as requirements_dir,
        ):
            doc_dir = Path(documentation_dir)
            license_file = doc_dir / "licenses" / "license.txt"
            license_file.parent.mkdir()
            license_file.write_text("Dummy license")

            # Requirements file creation
            req_dir = Path(requirements_dir)
            requirements_file = req_dir / "requirements.txt"
            existing_requirements = None

            for idx, file in enumerate(project_files):
                file_path = Path(file) if isinstance(file, str) else file
                if file_path.name == "requirements.txt":
                    project_files.pop(idx)
                    existing_requirements = file

            requirements = Model.ensure_requirements(existing_requirements, requirements_file)
            project_files.append(requirements)

            index_file = doc_dir / "index.html"
            index_file.write_text("dummy index")
            asBuilt = FmuBuilder.build_FMU(
                script,
                project_files=project_files,
                dest=dest,
                documentation_folder=doc_dir,
            )  # , xFunc=None)
            return asBuilt

    def to_xml(self, model_options: dict | None = None) -> ET.Element:
        """Build the XML FMI2 modelDescription.xml tree. (adapted from PythonFMU).

        Args:
            model_options ({str, str}) : Dict of FMU model options

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
            for a in self.default_experiment:  # check the dict
                assert a in (
                    "startTime",
                    "stopTime",
                    "stepSize",
                    "tolerance",
                ), f"DefaultExperiment key {a} unknown"
            ET.SubElement(
                root,
                "DefaultExperiment",
                {k: str(v) for k, v in self.default_experiment.items()},
            )

        variables = self._xml_modelvariables()
        root.append(variables)  # append <ModelVariables>

        structure = ET.SubElement(root, "ModelStructure")
        structure.append(self._xml_structure_outputs())
        initialunknowns = self._xml_structure_initialunknowns()
        if len(initialunknowns):
            structure.append(initialunknowns)
        return root

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
            for du in self._units[u]:  # list also the displays (if defined)
                unit.append(
                    ET.Element(
                        "DisplayUnit",
                        {
                            "name": du[0],
                            "factor": str(du[1](1.0)),
                            "offset": str(du[1](0.0)),
                        },
                    )
                )
            defs.append(unit)
        return defs

    def _xml_default_experiment(self):
        if self.default_experiment is not None:
            attrib = dict()
            if self.default_experiment.start_time is not None:
                attrib["startTime"] = str(self.default_experiment.start_time)
            if self.default_experiment.stop_time is not None:
                attrib["stopTime"] = str(self.default_experiment.stop_time)
            if self.default_experiment.step_size is not None:
                attrib["stepSize"] = str(self.default_experiment.step_size)
            if self.default_experiment.tolerance is not None:
                attrib["tolerance"] = str(self.default_experiment.tolerance)
        de = ET.Element("DefaultExperiment", attrib)
        return de

    def _xml_modelvariables(self):
        """Generate the FMI2 modelDescription.xml sub-tree <ModelVariables>."""
        mv = ET.Element("ModelVariables")
        for var in self.vars_iter():
            els = var.xml_scalarvariables()
            for el in els:
                mv.append(el)
        return mv

    def _xml_structure_outputs(self):
        """Generate the FMI2 modelDescription.xml sub-tree <ModelStructure><Outputs>.
        Exactly all variables with causality='output' must be in this list.
        .. todo:: implement support for variable dependencies and add as attribute here.
        """
        out = ET.Element("Outputs")

        for v in filter(
            lambda v: v is not None and v.causality == Causality.output,
            self.vars.values(),
        ):
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

        .. todo:: implement support for states and derivatives and add as attribute here.
        .. todo:: implement support for variable dependencies and add as attribute here.
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

        The iterator yields variable objects
        """
        if key is None:  # all variables
            for v in self.vars.values():
                if v is not None:
                    yield v
        elif isinstance(key, type):  # variable type iterator
            for v in self.vars.values():
                if v is not None and v.typ == key:
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

    def ref_to_var(self, vr: int):
        """Find Variable and sub-index (for compound variable), based on a value_reference value."""
        _vr = vr
        while True:
            var = self.vars[_vr]
            if var is None:
                _vr -= 1
            else:  # found the base of the variable
                return (var, vr - _vr)

    def _var_iter(self, vrs: list[int] | tuple[int, ...]):
        """Convert a list of value_reference integers into a variable object iterator.
        Take also compound variables into account:
          If all variables are listed, include the compound object in the result.
          Otherwise include the compound object and an index.
        """
        it = enumerate(vrs.__iter__())  # used also below!
        for i, vr in it:  # get an enumerated iterator over vrs
            sub = None
            assert vr in self.vars, f"Variable with valueReference={vr} does not exist in model {self.name}"
            var = self.vars[vr]
            if var is None:  # element of compound variable
                var, sub = self.ref_to_var(vr)
            elif isinstance(var, Variable) and len(var) > 1:
                sub = 0
            # At this point we should have a variable object
            if (
                isinstance(var, Variable)
                and len(var) > 1
                and i + len(var) <= len(vrs)
                and vr + len(var) - 1 == vrs[i + len(var) - 1]
            ):  # compound variable and all elements included
                for _ in range(len(var) - 1):  # spool to the last element
                    i, vr = next(it)
                sub = None
            # print(f"VAR_ITER whole {var.name} at {vr}")
            # print(f"_VAR_ITER {var.name}[{sub}], i:{i}, vr:{vr}")
            yield (var, sub)

    def _get(self, vrs: list[int] | tuple[int, ...], typ: type) -> list:
        """Get variables of all types based on references.
        This method is called by get_xxx and translates to fmi2GetXxx.
        """
        values = list()
        for var, sub in self._var_iter(vrs):
            check = var.typ == typ or (typ is int and issubclass(var.typ, Enum))
            assert check, f"Invalid type in 'get_{typ}'. Found variable {var.name} with type {var.typ}"
            val = var.getter()
            if var is not None and len(var) > 1:
                if sub is None:
                    values.extend(val)
                else:
                    values.append(val[sub])
            else:
                values.append(val)
        return values

    def get_integer(self, vrs: list[int] | tuple[int, ...]):
        return self._get(vrs, int)

    def get_real(self, vrs: list[int] | tuple[int, ...]):
        return self._get(vrs, float)

    def get_boolean(self, vrs: list[int] | tuple[int, ...]):
        return self._get(vrs, bool)

    def get_string(self, vrs: list[int] | tuple[int, ...]):
        return self._get(vrs, str)

    def _set(self, vrs: list[int] | tuple[int, ...], values: list | tuple, typ: type):
        """Set variables of all types. This method is called by set_xxx and translates to fmi2SetXxx.
        Variable range check, unit check and type check are performed by setter() function.
        on_set (if defined) is only run if the whole variable (all elements) are set.
        """
        idx = 0
        for var, sub in self._var_iter(vrs):
            check = var.typ == typ or (typ is int and issubclass(var.typ, Enum))
            assert check, f"Invalid type in 'set_{typ}'. Found variable {var.name} with type {var.typ}"
            if var is not None and len(var) > 1:
                if sub is None:  # set the whole vector
                    var.setter(values[idx : idx + len(var)], idx=None)
                    idx += len(var) - 1
                else:
                    var.setter(values[idx], sub)
            else:  # simple Variable
                var.setter(values[idx], idx=None)
            idx += 1

    def set_integer(self, vrs: list[int] | tuple[int, ...], values: list | tuple):
        self._set(vrs, values, int)

    def set_real(self, vrs: list[int] | tuple[int, ...], values: list | tuple):
        self._set(vrs, values, float)

    def set_boolean(self, vrs: list[int] | tuple[int, ...], values: list | tuple):
        self._set(vrs, values, bool)

    def set_string(self, vrs: list[int] | tuple[int, ...], values: list | tuple):
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
