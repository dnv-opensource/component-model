import logging
from enum import Enum, EnumType

import pytest
from pythonfmu.enums import Fmi2Causality as Causality
from pythonfmu.enums import Fmi2Initial as Initial
from pythonfmu.enums import Fmi2Variability as Variability

from component_model.enums import check_causality_variability_initial, combination, combinations, ensure_enum
from component_model.variable_naming import VariableNamingConvention

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def test_enum():
    def enum_func(e: Enum) -> None:
        assert isinstance(e, Enum), f"Argument {e} should be an enum member"
        logger.info(f"Name:{e.name}, value:{e.value}")

    def enumtype_func(e: EnumType):
        assert isinstance(e, EnumType), f"Argument {e} should be an EnumType, i.e. the Enum Class itself"
        logger.info(f"Members:{e._member_names_}")
        if "flat" in e.__members__:
            m: Enum = e["flat"]  # type: ignore[reportAssignmentType]
            logger.info(f"Member flat: {m}")

    f = VariableNamingConvention.flat
    assert isinstance(f, Enum)
    assert type(f) is VariableNamingConvention
    assert type(f)["structured"] == VariableNamingConvention.structured
    logger.info(f"Type of Enum class itself:{type(VariableNamingConvention)}")
    logger.info(f"Type of member:{type(VariableNamingConvention.flat)}")
    enum_func(VariableNamingConvention.flat)
    enumtype_func(VariableNamingConvention)


def test_combinations():
    assert (len(combinations), len(combinations[0])) == (5, 6)
    assert combination(Variability.discrete, Causality.output) == "C"
    assert combination(Variability.fixed, Causality.calculatedParameter) == "B"


def test_ensure_enum():
    assert ensure_enum("input", Causality.parameter) == Causality.input
    with pytest.raises(Exception) as err:
        ensure_enum("input", Variability.constant)
    assert str(err.value).startswith("The value input is not compatible with ")
    assert ensure_enum("discrete", Variability.continuous) == Variability.discrete
    assert ensure_enum("input", Causality.local) == Causality.input
    assert ensure_enum(None, Causality.input) == Causality.input, f"Found {ensure_enum(None, Causality.input)}"


def test_check():
    """
    # causality:
    # parameter, c.par, input, output, local, independent
    #                                    Variability
    ("a", "a", "a", "A", "A", "c"),  # constant
    ("A", "B", "d", "e", "B", "c"),  # fixed
    ("A", "B", "d", "e", "B", "c"),  # tunable
    ("b", "b", "D", "C", "C", "c"),  # discrete
    ("b", "b", "D", "C", "C", "E"),  # continuous
    """
    expected = (Causality.parameter, Variability.fixed, Initial.exact)
    assert check_causality_variability_initial("parameter", "fixed", "exact") == expected
    assert check_causality_variability_initial("parameter", "fixed", None) == expected
    assert check_causality_variability_initial("input", "fixed", None) == (None, None, None)
    assert check_causality_variability_initial("input", "discrete", Initial.approx) == (
        None,
        None,
        None,
    )
    assert check_causality_variability_initial("input", "discrete", None) == (
        Causality.input,
        Variability.discrete,
        None,
    )


if __name__ == "__main__":
    retcode = pytest.main(["-rA", "-v", __file__])
    assert retcode == 0, f"Non-zero return code {retcode}"
    # test_enum()
    # test_combinations()
    # test_ensure_enum()
    # test_check()
