"""Miscelaneous functions, not generally needed to make a FMU model,
but which can be useful to work with fmu files (e.g. retrieving and working with a modelDefinition.xml file),
to make an OSP system structure file, or to reverse-engineer the interface of a model
(e.g. when making a surrogate model).


"""

import xml.etree.ElementTree as ET  # noqa: N817
from enum import Enum
from pathlib import Path
from typing import Any
from zipfile import BadZipFile, ZipFile, is_zipfile


def xml_to_python_val(val: str):
    """Translate the xml (string) value to a python value and type."""
    if val == "true":
        return True
    elif val == "false":
        return False
    else:
        try:
            return int(val)
        except Exception:
            try:
                return float(val)
            except Exception:
                return {"Real": float, "Integer": int, "Boolean": bool, "String": str, "Enumeration": Enum}.get(
                    val, val
                )


def read_xml(xml: Path | str, sub: str = "modelDescription.xml") -> ET.Element:
    """Read xml file and return `sub` as Element object.

    xml can be

    * a zip file containing the xml file as `sub`
    * a xml file (e.g. modelDescription.xml).
    * a xml literal string. `sub` ignored in this case.
    """
    path = Path(xml)
    el = None
    if path.exists():  # we have a zip file or an xml file
        if is_zipfile(path):
            assert len(sub), "Information on file within zip needed"
            try:
                with ZipFile(path) as zp:
                    xml_string = zp.read(sub).decode()
            except Exception:
                raise BadZipFile(f"Not able to read zip file {xml} or {sub} not found in zipfile") from None
            el = ET.fromstring(xml_string)
        else:
            try:
                el = ET.parse(path).getroot()  # try to read the file directly, assuming a modelDescription.xml file
            except Exception:
                raise AssertionError(f"Could not parse xml file {path}") from None
    elif Path(path, sub).exists():  # unzipped xml path was provided
        try:
            el = ET.parse(Path(path, sub)).getroot()
        except ET.ParseError:
            raise AssertionError(f"Could not parse xml file {Path(path,sub)}") from None
    elif isinstance(xml, str):
        try:
            el = ET.fromstring(xml)  # try as literal string
        except ET.ParseError as err:
            raise AssertionError(
                f"Error when parsing {xml} as xml file. Error code {err.code} at {err.position}"
            ) from err
    else:
        raise Exception(f"Not possible to read model description from {xml}, {sub}") from None
    assert el is not None, f"xml {xml} not found or {sub} could not be read"
    return el


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
        models (dict)={}: dict of models (in OSP called 'simulators').
          A model is represented by a dict element modelName : {property:prop, variable:value, ...}
        connections (tuple)=(): tuple of model connections.
          Each connection is defined through a tuple of (model, variable, model, variable),
          where variable can be a tuple defining a variable group
        version (str)='0.1': The version of the system model
        start (float)=0.0: The simulation start time
        base_step (float)=0.01: The base stepSize of the simulation. The exact usage depends on the algorithm chosen
        algorithm (str)='fixedStep': The name of the algorithm

        .. todo:: better stepSize control in dependence on algorithm selected, e.g. with fixedStep we should probably set all step sizes to the minimum of everything?
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
        if isinstance(v1, (tuple, list)):  # group connection (e.g. a compound Variable)
            if not isinstance(v2, (tuple, list)) or len(v2) != len(v1):
                raise Exception(
                    f"Something wrong with the vector connection between {m1} and {m2}. Variable vectors do not match."
                ) from None
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


def model_from_fmu(fmu: str | Path, provideMsg: bool = False, sep=".") -> dict:
    """Generate a Model from an FMU (excluding the inner working functions like `do_step()`),
    i.e. partially reverse-engineering a FMU.
    This can be useful for convenient access to model information like variables
    or as starting point for surrogate models (e.g. speeding up models for optimisation studies
    or when using continuous time models in discrete event simulations)
    Note: structured variables with name: <name>[i], with otherwise equal causality, variability, initial
    and consecutive index and valueReference are stored as one Variable.
    .. todo:: <UnitDefinitions>, <LogCategories>.

    Args:
        fmu (str, Path): the FMU file which is to be read. can be the full FMU zipfile, the modelDescription.xml or a equivalent string
        provideMsg (bool): Optional possibility to provide messages during the process (for debugging purposes)
        sep (str)='.': separation used for structured variables (both for sub-systems and variable names)

    Returns
    -------
        Arguments of Model.__init__ as dict (**not** the model object itself)
    """

    el = read_xml(fmu)
    _de = el.find(".//DefaultExperiment")
    de = {}
    if _de is not None:
        de["start_time"] = float(_de.get("start", 0.0))
        if "stopTime" in _de.attrib:
            de["stop_time"] = float(_de.attrib["stopTime"])
        if "stepSize" in _de.attrib:
            de["step_size"] = float(_de.attrib["stepSize"])
        if "tolerance" in _de.attrib:
            de["tolerance"] = float(_de.attrib["tolerance"])

    co_flags = el.find(".//CoSimulation")
    flags = {} if co_flags is None else {key: xml_to_python_val(val) for key, val in co_flags.attrib.items()}
    kwargs: dict[str, Any] = {}
    kwargs["name"] = el.attrib["modelName"]
    kwargs["description"] = el.get("description", f"Component model object generated from {fmu}")
    kwargs["author"] = el.get("author", "anonymous")
    kwargs["version"] = el.get("version", "0.1")
    kwargs["unit_system"] = "SI"
    kwargs["license"] = el.get("license", "")
    kwargs["copyright"] = el.get("copyright", "")
    kwargs["guid"] = el.get("guid", "")
    kwargs["default_experiment"] = de
    kwargs["flags"] = (flags,)
    return kwargs


def variables_from_fmu(el: ET.Element | None, sep: str = "["):
    """From the supplied <ModelVariables> el subtree identify and define all variables.
    Return an iterator through variable arguments as dict,
    so that variables can be added to model through `.add_variable(**kwargs)`.
    .. toDo:: implement unit and displayUnit handling + <UnitDefinitions>.

    Returns
    -------
        List of argument dicts, which then can be used to instantiate Variable objects `Variable(**kwargs)`
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
            base, _sub = txt.rsplit(sep, maxsplit=1)
            pair2 = {"[": "]", "(": ")", "{": "}"}[sep]
            _sub = _sub.rsplit(pair2, maxsplit=1)[0]
            try:
                sub = int(_sub)
            except ValueError:
                sub = None
            return (base, sub)
        else:
            return (txt, None)

    def get_start(v, _typ):
        start = v.attrib.get("start", None)
        if start is None:  # set an 'example' value of the correct type
            start = {float: 1.0, int: 1, bool: False, str: "", Enum: 1}[_typ]
        return start

    def get_basic_kwargs(sv):
        """Get the basic kwargs of this ScalarVariable,
        i.e. the kwargs which are the same for all elements of compound variables.
        """
        base, sub = rsplit_sep(sv.attrib["name"])
        kwa = {}
        kwa["name"] = base
        kwa["causaity"] = var.get("causality", "local")
        kwa["variability"] = (var.get("variability", "continuous"),)
        kwa["initial"] = (var.get("initial", None),)
        kwa["_typ"] = xml_to_python_val(sv[0].tag)
        return (kwa, sub)

    idx = 0
    while el is not None and len(el) and idx < len(el) - 1:  # type: ignore
        var = el[idx]
        kwargs, sub = get_basic_kwargs(var)
        start = [get_start(var[0], kwargs["_typ"])]
        rng = [range_from_fmu(var[0])]
        # check whether the next variables are elements of the same:
        if sub == 0:
            for i in range(1, len(el) - idx):
                v = el[idx + i]
                kwa, s = get_basic_kwargs(v)
                if all(kwargs[x] == kwa[x] for x in kwa) and s == sub + i:  # next element
                    start.append(get_start(v[0], kwargs["_typ"]))
                    rng.append(range_from_fmu(v[0]))
                else:
                    break
            idx += i  # jump the compound elements
        else:
            assert sub is None, f"Unclear whether variable {var.get('name', '??')} is compound"
            idx += 1
        kwargs.update(
            {
                "valueReference": int(var.get("valueReference", -1)),
                "start": start[0] if len(start) == 1 else tuple(start),
                "rng": rng[0] if len(rng) == 1 else tuple(rng),
            }
        )
        yield kwargs
