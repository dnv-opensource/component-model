import logging

import pytest
from pythonfmu.enums import Fmi2Causality as Causality  # type: ignore
from pythonfmu.enums import Fmi2Initial as Initial  # type: ignore
from pythonfmu.enums import Fmi2Variability as Variability  # type: ignore

import component_model.caus_var_ini as cvi

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def test_combinations():
    assert (len(cvi.combinations), len(cvi.combinations[0])) == (5, 6)
    assert cvi.combination(Variability.discrete, Causality.output) == "C"
    assert cvi.combination(Variability.fixed, Causality.calculatedParameter) == "B"


def test_ensure_enum():
    assert cvi.ensure_enum("input", Causality, Causality.parameter) == Causality.input
    with pytest.raises(Exception) as err:
        cvi.ensure_enum("input", Variability, Causality.output)
    assert str(err.value).startswith("The value input is not compatible with ")
    assert cvi.ensure_enum("discrete", Variability, None) == Variability.discrete
    assert cvi.ensure_enum("input", Causality, None) == Causality.input
    assert cvi.ensure_enum(None, Causality, Causality.input) == Causality.input


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
    assert cvi.check_causality_variability_initial("parameter", "fixed", "exact") == expected
    assert cvi.check_causality_variability_initial("parameter", "fixed", None) == expected
    assert cvi.check_causality_variability_initial("input", "fixed", None) == (None, None, None)
    assert cvi.check_causality_variability_initial("input", "discrete", Initial.approx) == (
        None,
        None,
        None,
    )
    assert cvi.check_causality_variability_initial("input", "discrete", None) == (
        Causality.input,
        Variability.discrete,
        None,
    )


if __name__ == "__main__":
    retcode = pytest.main(["-rA", "-v", __file__])
    assert retcode == 0, f"Non-zero return code {retcode}"
    # test_ensure_enum()
