import logging

import pytest

from component_model.variable_naming import ParsedVariable, VariableNamingConvention

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


test_cases = [
    ("vehicle.engine.speed", ("vehicle.engine", "speed", [], 0)),
    ("resistor12.u", ("resistor12", "u", [], 0)),
    ("v_min", (None, "v_min", [], 0)),
    ("robot.axis.'motor #234'", ("robot.axis", "'motor #234'", [], 0)),
    ("der(pipe[3,4].T[14],2)", ("pipe[3,4]", "T", [14], 2)),
    ("der(pipe[3,4].T[14])", ("pipe[3,4]", "T", [14], 1)),
    ("pipe[3,4].T[14]", ("pipe[3,4]", "T", [14], 0)),
    ("T[14].pipe[3,4]", ("T[14]", "pipe", [3, 4], 0)),
    ("der(pos,1)", (None, "pos", [], 1)),
    ("der(wheels[0].motor.rpm)", ("wheels[0].motor", "rpm", [], 1)),
]


@pytest.mark.parametrize("txt, expected", test_cases)
def test_basic_re_expressions(txt, expected):
    """Test the expressions used in variable_naming."""
    parsed = ParsedVariable(txt, VariableNamingConvention.structured)
    tpl = (parsed.parent, parsed.var, parsed.indices, parsed.der)
    for i in range(4):
        assert tpl[i] == expected[i], f"Test:{txt}. Variable element {i} {tpl[i]} != {expected[i]}"
    # print(tpl)


if __name__ == "__main__":
    retcode = 0  # pytest.main(["-rA", "-v", __file__])
    assert retcode == 0, f"Non-zero return code {retcode}"
    for c, e in test_cases:
        print(c, e)
        test_basic_re_expressions(c, e)
