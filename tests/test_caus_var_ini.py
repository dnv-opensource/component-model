from enum import Enum

import component_model.caus_var_ini as cvi  # type: ignore
import pytest
from pythonfmu.enums import Fmi2Causality as Causality  # type: ignore
from pythonfmu.enums import Fmi2Variability as Variability  # type: ignore


def test_combinations():
    assert (len(cvi.combinations), len(cvi.combinations[0])) == (5, 6)
    assert cvi.combination(Variability.discrete, Causality.output) == "C"
    assert cvi.combination(Variability.fixed, Causality.calculatedParameter) == "B"


def test_ensure_enum():
    assert cvi.ensure_enum("input", Causality, Causality.parameter) == Causality.input
    with pytest.raises(Exception) as err:
        cvi.ensure_enum("input", cvi.Initial, Causality.output)
    assert str(err.value).startswith("The value input is not compatible with ")
    assert cvi.ensure_enum("discrete", Variability, None) == Variability.discrete
    assert cvi.ensure_enum("input", Causality, None) == Causality.input
    assert cvi.ensure_enum(None, Causality, Causality.input) == Causality.input


def test_check():
    assert cvi.check_causality_variability_initial("input", "discrete", cvi.Initial.approx, msg=True) == (None, None, None)
    assert cvi.check_causality_variability_initial("input", "discrete", None, msg=True) == (
        Causality.input,
        Variability.discrete,
        cvi.Initial.none,
    )
