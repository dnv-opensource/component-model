from pathlib import Path

import pytest
from component_model.model import Model
from component_model.utils import (
    make_osp_system_structure,
    model_from_fmu,
    read_xml,
    variables_from_fmu,
    xml_to_python_val,
)


@pytest.fixture(scope="session")
def bouncing_ball_fmu():
    build_path = Path.cwd() / "fmus"
    build_path.mkdir(exist_ok=True)
    fmu_path = Model.build(
        str(Path(__file__).parent.parent / "component_model" / "example_models" / "bouncing_ball.py"),
        project_files=[],
        dest=build_path,
    )
    return fmu_path


def dicts_equal(d1: dict, d2: dict):
    assert isinstance(d1, dict), f"Dict expected. Found {d1}"
    assert isinstance(d2, dict), f"Dict expected. Found {d2}"
    for key in d1:
        assert key in d2, f"Key {key} not found in {d2}"
        assert d1[key] == d2[key], f"Value of key {key} {d1[key]} != {d2[key]}"
    for key in d2:
        assert key in d1, f"Key {key} not found in {d1}"
        assert d1[key] == d2[key], f"Value of key {key} {d1[key]} != {d2[key]}"


def test_xml_to_python_val():
    assert xml_to_python_val("true")
    assert not xml_to_python_val("false")
    assert xml_to_python_val("99") == 99, "Detect an int"
    assert xml_to_python_val("99.99") == 99.99, "Detect a float"
    assert xml_to_python_val("Real") == float, "Detect a type"
    assert xml_to_python_val("String") == str, "Detect a type"
    assert xml_to_python_val("Real") == float, "Detect a type"
    assert xml_to_python_val("Hello World") == "Hello World", "Detect a literal string"


def test_model_description(bouncing_ball_fmu):
    et = read_xml(bouncing_ball_fmu)
    assert et is not None, "No Model Description"
    for a in (
        "fmiVersion",
        "modelName",
        "guid",
        "generationTool",
        "generationDateAndTime",
        "variableNamingConvention",
        "description",
        "author",
        "license",
    ):
        assert a in et.attrib, f"Attribute fmiModeldescription: {a} not found"
    el = et.find("./CoSimulation")
    assert el is not None, "CoSimulation element expected"
    dicts_equal(
        el.attrib,
        {
            "needsExecutionTool": "true",
            "canHandleVariableCommunicationStepSize": "true",
            "canInterpolateInputs": "false",
            "canBeInstantiatedOnlyOncePerProcess": "false",
            "canGetAndSetFMUstate": "false",
            "canSerializeFMUstate": "false",
            "modelIdentifier": "BouncingBall",
            "canNotUseMemoryManagementFunctions": "true",
        },
    )
    assert el.find("./SourceFiles") is not None, "SourceFiles expected"
    el = et.find("./UnitDefinitions")
    assert el is not None, "UnitDefinitions element expected"
    assert (
        len(el) == 5
    ), f"Five UnitDefinitions expected. Found {''.join(x.get('name')+', ' for x in el.findall('./Unit'))}"
    el = et.find("./TypeDefinitions")
    assert el is None, "No TypeDefinitions expected (so far not implemented in component_model"
    el = et.find("./LogCategories")
    assert el is not None, "LogCategory element expected"
    assert (
        len(el) == 5
    ), f"Five LogCategories expected. Found {''.join(x.get('name')+', ' for x in el.findall('./Category'))}"
    el = et.find("./DefaultExperiment")
    assert el.attrib == {"startTime": "0", "stopTime": "1.0", "stepSize": "0.01"}, f"DefaultExperiment: {el.attrib}"
    el = et.find("./ModelVariables")
    assert el is not None, "ModelVariables element expected"
    assert (
        len(el) == 11
    ), f"11 ModelVariables expected. Found {''.join(x.get('name')+', ' for x in el.findall('./ScalarVariable'))}"
    el = et.find("./ModelStructure")
    assert el is not None, "ModelStructure element expected"
    e = el.find("./Outputs")
    assert e is not None, "Outputs element expected"
    assert len(e) == 9, f"9 Outputs expected. Found {''.join(x.get('index')+', ' for x in e.findall('./Unknown'))}"
    e = el.find("./InitialUnknowns")
    assert e is not None, "InitialUnknowns element expected"
    assert (
        len(e) == 3
    ), f"3 InitialUnknowns expected. Found {''.join(x.get('index')+', ' for x in e.findall('./Unknown'))}"


def test_osp_structure(tmp_path_factory):
    make_osp_system_structure(
        str(tmp_path_factory.mktemp("fmu") / "systemModel"),
        version="0.1",
        models={
            "simpleTable": {"interpolate": True},
            "mobileCrane": {"pedestal.pedestalMass": 5000.0, "boom.boom.0": 20.0},
        },
        connections=("simpleTable", "outputs.0", "mobileCrane", "pedestal.angularVelocity"),
    )


def test_model_from_fmu(bouncing_ball_fmu):
    kwargs = model_from_fmu(bouncing_ball_fmu)
    kwargs.pop("guid")
    expected = {
        "name": "BouncingBall",
        "description": "Another BouncingBall model, made in Python and using Model and Variable to construct a FMU",
        "author": "DNV, SEACo project",
        "version": "0.1",
        "unit_system": "SI",
        "license": 'Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. ',
        "copyright": "Copyright (c) 2024 DNV, SEACo project",
        "default_experiment": {"start_time": 0.0, "stop_time": 1.0, "step_size": 0.01},
        "flags": (
            {
                "needsExecutionTool": True,
                "canHandleVariableCommunicationStepSize": True,
                "canInterpolateInputs": False,
                "canBeInstantiatedOnlyOncePerProcess": False,
                "canGetAndSetFMUstate": False,
                "canSerializeFMUstate": False,
                "modelIdentifier": "BouncingBall",
                "canNotUseMemoryManagementFunctions": True,
            },
        ),
    }
    dicts_equal(kwargs, expected)


def test_variables_from_fmu(bouncing_ball_fmu):
    et = read_xml(bouncing_ball_fmu)
    mv = et.find(".//ModelVariables")
    collect = []
    for kwargs in variables_from_fmu(mv):
        collect.append(kwargs)
    assert len(collect) == 5
    assert collect[0]["name"] == "pos"
    assert len(collect[1]["start"]) == 3
    assert len(collect[4]["rng"][0]) == 0
