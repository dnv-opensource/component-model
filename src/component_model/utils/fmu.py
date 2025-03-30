"""Miscelaneous functions, not generally needed to make a FMU model,
but which can be useful to work with fmu files (e.g. retrieving and working with a modelDefinition.xml file),
to make an OSP system structure file, or to reverse-engineer the interface of a model
(e.g. when making a surrogate model).


"""

import xml.etree.ElementTree as ET  # noqa: N817
from enum import Enum
from pathlib import Path
from typing import Any

from component_model.utils.xml import read_xml, xml_to_python_val


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
            i = 0
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
