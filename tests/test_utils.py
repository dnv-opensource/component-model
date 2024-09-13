from pathlib import Path

from component_model.model import Model
from component_model.utils import (
    make_osp_system_structure,
    read_model_description,
    variables_from_fmu,
    xml_to_python_val,
)


def ensure_bouncing_ball():
    fmu = "BouncingBall.fmu"
    if not Path("./" + fmu).exists():
        fmu = Model.build("../component_model/example_models/bouncing_ball.py")
    return fmu


def test_xml_to_python_val():
    assert xml_to_python_val("true")
    assert not xml_to_python_val("false")
    assert xml_to_python_val("99") == 99, "Detect an int"
    assert xml_to_python_val("99.99") == 99.99, "Detect a float"
    assert xml_to_python_val("Real") == float, "Detect a type"
    assert xml_to_python_val("String") == str, "Detect a type"
    assert xml_to_python_val("Real") == float, "Detect a type"


#    def test_model_description(self):
def test_osp_structure():
    make_osp_system_structure(
        "systemModel",
        version="0.1",
        models={
            "simpleTable": {"interpolate": True},
            "mobileCrane": {"pedestal.pedestalMass": 5000.0, "boom.boom.0": 20.0},
        },
        connections=("simpleTable", "outputs.0", "mobileCrane", "pedestal.angularVelocity"),
    )


def test_variables_from_fmu():
    fmu = ensure_bouncing_ball()
    et = read_model_description(fmu)
    mv = et.find(".//ModelVariables")
    collect = []
    for kwargs in variables_from_fmu(mv):
        collect.append(kwargs)
    assert len(collect) == 5
    assert collect[0]["name"] == "pos"
    assert len(collect[1]["start"]) == 3
    assert len(collect[4]["rng"][0]) == 0


if __name__ == "__main__":
    #    retcode = pytest.main(["-rP -s -v", __file__])
    #    assert retcode == 0, f"Return code {retcode}"
    test_xml_to_python_val()
    test_variables_from_fmu()
