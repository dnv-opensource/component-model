import logging
import time
import xml.etree.ElementTree as ET  # noqa: N817
from pathlib import Path

import pytest

from component_model.model import Model  # type: ignore
from component_model.utils.fmu import model_from_fmu
from component_model.variable import Check, Variable
from component_model.variable_naming import ParsedVariable, VariableNamingConvention

logger = logging.getLogger(__name__)


class DummyModel(Model):
    def __init__(self, name, **kwargs):
        super().__init__(name=name, description="Just a dummy model to be able to do testing", **kwargs)

    def do_step(self, current_time: float, step_size: float):
        return True


@pytest.fixture(scope="session")
def bouncing_ball_fmu(tmp_path_factory):
    return _bouncing_ball_fmu()


def _bouncing_ball_fmu():
    build_path = Path.cwd()
    build_path.mkdir(exist_ok=True)
    fmu_path = Model.build(
        Path(__file__).parent.parent / "examples" / "bouncing_ball_3d.py",
        project_files=[],
        dest=build_path,
    )
    return fmu_path


def test_license():
    mod = DummyModel("TestModel", author="Ola Norman")
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


def test_variable_naming():
    mod = DummyModel("TestModel", author="Ola Norman", variable_naming="flat")
    conv = VariableNamingConvention.flat
    assert mod.variable_naming == conv
    assert ParsedVariable("x", conv).as_tuple() == (None, "x", [], 0)
    assert ParsedVariable("x[3]", conv).as_tuple() == (None, "x", [3], 0)
    assert ParsedVariable("x[3,5]", conv).as_tuple() == (None, "x", [3, 5], 0)


def test_xml():
    Model.instances = []  # reset
    mod = DummyModel("MyModel")
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
    assert ET.tostring(el[0], encoding="unicode") == '<Unit name="meter"><BaseUnit m="1" factor="1" /></Unit>'
    expected = '<Unit name="radian"><BaseUnit rad="0" factor="1" /><DisplayUnit name="degree" factor="0.017453292519943292" offset="0.0" /></Unit>'
    assert ET.tostring(el[1], encoding="unicode") == expected, f"Found {ET.tostring(el[1], encoding='unicode')}"
    expected = '<Unit name="dimensionless"><BaseUnit factor="1" /><DisplayUnit name="percent" factor="0.01" offset="0.0" /></Unit>'
    assert ET.tostring(el[2], encoding="unicode") == expected, f"Found {ET.tostring(el[2], encoding='unicode')}"

    et = mod.to_xml()
    # check that all expected elements are in ModelDescription
    assert et.tag == "fmiModelDescription"
    assert et.find(".//UnitDefinitions") is not None
    assert et.find(".//UnitDefinitions/Unit") is not None
    assert et.find(".//LogCategories") is not None
    assert et.find(".//DefaultExperiment") is not None
    de = et.find(".//DefaultExperiment")
    assert (
        de is not None
        and ET.tostring(de) == b'<DefaultExperiment startTime="0.0" stopTime="1.0" stepSize="0.01" tolerance="0.001" />'
    )
    assert et.find(".//ModelVariables") is not None
    assert et.find(".//ModelVariables/ScalarVariable") is not None
    assert et.find(".//ModelStructure") is not None
    # print( ET.tostring(et))


def test_from_fmu(bouncing_ball_fmu):
    model = model_from_fmu(bouncing_ball_fmu)
    assert model["name"] == "BouncingBall3D", f"Name:{model['name']}"
    expected = "Another Python-based BouncingBall model, using Model and Variable to construct a FMU"
    assert model["description"] == expected, f"Description:{model['description']}"
    assert model["author"] == "DNV, SEACo project"
    assert model["version"] == "0.1"
    assert model["license"].startswith("Permission is hereby granted, free of charge, to any person obtaining a copy")
    assert model["copyright"] == f"Copyright (c) {time.localtime()[0]} DNV, SEACo project", (
        f"Found: {model['copyright']}"
    )
    assert model["default_experiment"] is not None
    assert (
        model["default_experiment"]["start_time"],
        model["default_experiment"]["step_size"],
        model["default_experiment"]["stop_time"],
    ) == (0.0, 0.01, 1.0)


if __name__ == "__main__":
    retcode = 0  # pytest.main(["-rA", "-v", __file__])
    assert retcode == 0, f"Non-zero return code {retcode}"
    # test_license()
    # test_xml()
    test_from_fmu(_bouncing_ball_fmu())
    # test_variable_naming()
