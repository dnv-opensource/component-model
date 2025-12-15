import logging

import pytest

from component_model.variable_naming import ParsedVariable, VariableNamingConvention

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def test_single():
    which = -1
    if which == -1 or which == 1:
        parsed = ParsedVariable("der(pos,1)", VariableNamingConvention.structured)
        assert parsed.as_string() == "der(pos)"
        assert parsed.as_tuple() == (None, "pos", [], 1)
        assert parsed.var == "pos"
    if which == -1 or which == 2:
        parsed = ParsedVariable("der(pos,2)", VariableNamingConvention.structured)
        assert parsed.as_string() == "der(pos,2)"
    if which == -1 or which == 3:
        parsed = ParsedVariable("der(der(pos,1))", VariableNamingConvention.structured)
        assert parsed.as_tuple() == (None, "pos", [], 2), f"Found {parsed.as_tuple()}"
        assert parsed.as_string(simplified=True, primitive=True) == "der(pos)"
    if which == -1 or which == 4:
        parsed = ParsedVariable("der(pipe[3,4].T[14])", VariableNamingConvention.structured)
        assert parsed.as_string(primitive=True, index="13") == "pipe[3,4].T[13]"


test_cases = [
    # to-parse               expected                            msg
    ("vehicle.engine.speed", ("vehicle.engine", "speed", [], 0), ""),
    ("resistor12.u", ("resistor12", "u", [], 0), ""),
    ("v_min", (None, "v_min", [], 0), ""),
    ("robot.axis.'motor #234'", ("robot.axis", "'motor #234'", [], 0), ""),
    ("der(pipe[3,4].T[14],2)", ("pipe[3,4]", "T", [14], 2), ""),
    ("der(pipe[3,4].T[14])", ("pipe[3,4]", "T", [14], 1), ""),
    ("pipe[3,4].T[14]", ("pipe[3,4]", "T", [14], 0), ""),
    ("T[14].pipe[3, 4]", ("T[14]", "pipe", [3, 4], 0), ""),
    ("der(pos)", (None, "pos", [], 1), ""),
    ("der(pos,1)", (None, "pos", [], 1), "der(pos)"),
    ("der(wheels[0].motor.rpm)", ("wheels[0].motor", "rpm", [], 1), ""),
    ("der(der(wheels[0].motor.rpm))", ("wheels[0].motor", "rpm", [], 2), "der(wheels[0].motor.rpm,2)"),
    ("der(wheels[0].motor.rpm,2)", ("wheels[0].motor", "rpm", [], 2), ""),
]


@pytest.mark.parametrize("txt, expected, text", test_cases)
def test_basic_re_expressions(txt, expected, text):
    """Test the expressions used in variable_naming."""
    parsed = ParsedVariable(txt, VariableNamingConvention.structured)
    tpl = (parsed.parent, parsed.var, parsed.indices, parsed.der)
    for i in range(4):
        assert tpl[i] == expected[i], f"Test:{txt}. Variable element {i} {tpl[i]} != {expected[i]}"
    expected_txt = txt if text == "" else text
    assert parsed.as_string() == expected_txt, f"as_string {parsed.as_tuple()}: {parsed.as_string()}. Expected:{txt}"


def _test_basic_re_expressions():
    for c, e, t in test_cases:
        print(c, e, t)
        test_basic_re_expressions(c, e, t)


def test_as_string():
    parsed = ParsedVariable("der(pipe[3,4].T[14], 2)", VariableNamingConvention.structured)
    assert parsed.as_string(("parent", "var")) == "pipe[3,4].T"
    assert parsed.as_string(("parent", "var", "indices")) == "pipe[3,4].T[14]"
    parsed = ParsedVariable("der(wheels[0].motor.rpm)")
    assert parsed.as_string(("parent", "var")) == "wheels[0].motor.rpm"
    assert parsed.as_string(("parent", "var", "indices")) == "wheels[0].motor.rpm"
    assert parsed.as_string(("parent", "var", "indices", "der"), simplified=False) == "der(wheels[0].motor.rpm,1)"


if __name__ == "__main__":
    retcode = pytest.main(["-rA", "-v", __file__])
    assert retcode == 0, f"Non-zero return code {retcode}"
    # test_single()
    # _test_basic_re_expressions()
    # test_as_string()
