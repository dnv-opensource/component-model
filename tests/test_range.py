import numpy as np
import pytest

from component_model.enums import Check
from component_model.range import Range
from component_model.unit import Unit


@pytest.fixture
def unt(scope: str = "module", autouse: bool = True):
    return _unt()


def _unt():
    _registry = Unit.ensure_unit_registry("SI")
    return Unit()


def test_init(unt: Unit):
    _rng = Range(1.0, (0, 2))
    int1 = Range(1, None)
    assert int1.rng == (1, 1), "Missing range. restricted to fixed start value."
    int2 = Range(1, rng=(0, 5))
    assert int2.rng == (0, 5), "That works"
    assert not int2.check(-1), "Error detected function"
    float1 = Range(1.0)
    assert float1.rng == (float("-inf"), float("inf")), "Auto_extreme. Same as rng=()"
    float2 = Range(1.0, rng=None)  # implicit type through start value and no range
    assert float2.rng == (1.0, 1.0), "No range."
    assert not float2.check(-1), "Error detected"
    float3 = Range(1.0, ("0m", "10m"), unit=Unit("1 m"))
    assert float3.rng == (0.0, 10.0), f"Found {float3.rng}"
    float4 = Range(0.55, ("0 %", None), unit=Unit("55%"))
    assert np.allclose(float4.rng, (0.0, 55.0)), f"Found {float4.rng} != {(0.0, 55.0)}"


def test_auto_extreme():
    """Test auto_extreme, defined for bool, float and Enum types."""
    assert Range.auto_extreme(1.0) == (float("-inf"), float("inf"))
    assert Range.auto_extreme(bool) == (False, True)
    assert Range.auto_extreme(Check.all) == (1, 2)
    assert Range.auto_extreme(Check) == (1, 2)
    with pytest.raises(ValueError) as err:
        Range.auto_extreme(1)
    assert str(err.value) == "Auto-extremes for type <class 'int'> cannot be determined"


if __name__ == "__main__":
    retcode = pytest.main(["-rP -s -v", __file__])
    assert retcode == 0, f"Return code {retcode}"
    # _unt() # initialize UnitRegistry (otherwise Unit cannot be used)
    # test_init(_unt())
    # test_auto_extreme()
