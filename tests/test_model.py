import logging
import time
import xml.etree.ElementTree as ET  # noqa: N817

import pytest
from component_model.logger import get_module_logger  # type: ignore
from component_model.model import Model  # type: ignore
from component_model.variable import Check, Variable

logger = get_module_logger(__name__, level=logging.INFO)


def test_license():
    mod = Model("TestModel", author="Ola Norman")
    c, lic = mod.make_copyright_license(None, None)
    assert c == f"Copyright (c) {time.localtime()[0]} Ola Norman", f"Found: {c}"
    assert lic.startswith("Permission is hereby granted, free of charge, to any person obtaining a copy")
    _lic = """Copyright (c) 2023 Ola Norman

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and a ..."""
    c, lic = mod.make_copyright_license(None, _lic)
    assert c == "Copyright (c) 2023 Ola Norman"
    assert lic.startswith("Permission is hereby granted, free of charge, to any person obtaining a copy")

    c, lic = mod.make_copyright_license("Copyleft (c) 3000 Nobody", _lic)
    assert c == "Copyleft (c) 3000 Nobody"
    assert lic.strip().startswith("Permission is hereby granted, free of charge, to any person obtaining a copy")


def test_xml():
    Model.instances = []  # reset
    mod = Model("MyModel")
    _ = Variable(
        mod,
        "Test9",
        description="A NP variable with units included in initial values and partially fixed range",
        causality="output",
        variability="continuous",
        start=("1m", "2deg", "3 deg"),
        rng=((0, "3m"), None, None),
    )
    _ = Variable(
        mod,
        "myInt",
        description="A integer variable",
        causality="parameter",
        variability="fixed",
        start="99%",
        rng=(0, "100%"),
        annotations=None,
        value_check=Check.all,
    )
    el = mod._xml_modelvariables()
    assert el.tag == "ModelVariables"
    assert len(el) == 4
    assert el[0].tag == "ScalarVariable" and el[0].get("name") == "Test9[0]"
    assert el[3].tag == "ScalarVariable" and el[3].get("name") == "myInt"
    el = mod._xml_structure_outputs()
    assert ET.tostring(el) == b'<Outputs><Unknown index="1" /><Unknown index="2" /><Unknown index="3" /></Outputs>'
    el = mod._xml_structure_initialunknowns()
    assert (
        ET.tostring(el)
        == b'<InitialUnknowns><Unknown index="1" /><Unknown index="2" /><Unknown index="3" /></InitialUnknowns>'
    )
    el = mod.xml_unit_definitions()
    assert el.tag == "UnitDefinitions"
    assert len(el) == 3
    # assert ET.tostring(el[0]) == b'<Unit name="dimensionless" />'
    assert ET.tostring(el[0]) == b'<Unit name="meter"><BaseUnit m="1" factor="1" /></Unit>'
    expected = b'<Unit name="radian"><BaseUnit rad="0" factor="1" /><DisplayUnit name="degree" factor="0.017453292519943292" offset="0.0" /></Unit>'
    assert ET.tostring(el[1]) == expected, f"Found {ET.tostring(el[1])}"
    expected = b'<Unit name="dimensionless"><BaseUnit factor="1" /><DisplayUnit name="percent" factor="0.01" offset="0.0" /></Unit>'
    assert ET.tostring(el[2]) == expected, f"Found {ET.tostring(el[2])}"

    et = mod.to_xml()
    # check that all expected elements are in ModelDescription
    assert et.tag == "fmiModelDescription"
    assert et.find(".//UnitDefinitions") is not None
    assert et.find(".//UnitDefinitions").find(".//Unit") is not None
    assert et.find(".//LogCategories") is not None
    assert et.find(".//DefaultExperiment") is not None
    assert (
        ET.tostring(et.find(".//DefaultExperiment"))
        == b'<DefaultExperiment startTime="0" stopTime="1.0" stepSize="0.01" />'
    )
    assert et.find(".//ModelVariables") is not None
    assert et.find(".//ModelVariables").find(".//ScalarVariable") is not None
    assert et.find(".//ModelStructure") is not None
    # print( ET.tostring(et))


def test_from_fmu():
    path = Path(Path(__file__).parent, "BouncingBallFMU.fmu")
    assert path.exists(), "FMU not found"
    model = model_from_fmu( path)
    assert model.name == "BouncingBallFMU", f"Name:{model.name}"
    print( dir(model))
    assert model.description == "Simple bouncing ball test FMU", f"Description:{model.description}"
    assert model.author == "DNV, SEACo project"
    assert model.version == "0.1"
    assert model.license.startswith("Permission is hereby granted, free of charge, to any person obtaining a copy")
    assert model.copyright == f"Copyright (c) {time.localtime()[0]} DNV, SEACo project", f"Found: {model.copyright}"
    assert model.guid == "8336c04d3e2f45379db8ed685e034a69", f"Found: {model.guid}"
    assert model.default_experiment is None
    assert (
        model.default_experiment.start_time,
        model.default_experiment.step_size,
        model.default_experiment.stop_time,
        model.default_experiment.tolerance,
    ) == (0.0, 0.1, 10.0, 0.001)
    assert model.flags == {
        "needsExecutionTool": True,
        "canHandleVariableCommunicationStepSize": True,
        "canNotUseMemoryManagementFunctions": True,
    }
    for idx, var in model.vars.items():
        print(idx, var)
    assert model.vars[0].name == "x[0]"
    assert model.vars[0].value0 == 0.0
    assert model.vars[6].name == "bounceFactor"

