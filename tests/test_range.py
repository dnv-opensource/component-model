import logging
import numpy as np
import pytest
from enum import Enum
from typing import Any, Sequence, Never

from component_model.enums import Check
from component_model.range import Range
from component_model.unit import Unit
from component_model.enums import Check

logger = logging.getLogger(__name__)


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


def test_range_spec():
    def do_check(example:Any,
                 rng : tuple[Any,...] | tuple[Any,Any] | None | Sequence[Never],
                 var_len : int,
                 typ: type,
                 level: int=0,
                 expect:int=0,
                 msg:str = "")-> int:
        ck = Range.is_valid_spec( rng, var_len, typ, level)
        if ck == 0:
            if var_len==1:
                try:
                    _rng = Range(example, rng)
                except ValueError as err:
                    logger.error(f"{rng} set to valid, but {err}")
            else:
                for e,r in zip(example,rng, strict=True):
                    try:
                        _rng = Range(e, r)
                    except ValueError as err:
                        logger.error(f"{r} set to valid, but {err}")
        assert ck == expect, f"{msg}: Range:{rng}: {ck} != {expect}"
    
    Unit.ensure_unit_registry()
    do_check( 9.9, tuple(), 1, float, expect=0, msg="Valid single variable spec with automatic floats.")
    do_check( 9, tuple(), 1, int, expect=1, msg="InValid single variable spec with automatic int.")
    do_check( 9, None, 1, int, expect=0, msg="Valid fixed single variable spec (any type).")
    do_check( Check.all, None, 1, Enum, expect=0, msg="Valid fixed single variable spec (any type).")
    do_check( Check.all, tuple(), 1, Enum, expect=0, msg="Valid automatic enum single variable range.")
    do_check( 1, (1,2), 1, int, expect=0, msg="Valid range of single int variable.")
    do_check( 5.0, (9.9,2.0), 1, float, expect=19, msg="InValid range of single float variable: wrong order.")
    do_check( 1.0, ("1m",2.0), 1, float, expect=0, msg="Valid range of single float variable (partial units).")
    do_check( (1.0,2.0,3.0), (None,tuple(),(1.0,2.0)), 3, float, expect=0, msg="Valid mixed vector spec.")
    

if __name__ == "__main__":
    retcode = 0#pytest.main(["-rP -s -v", __file__])
    assert retcode == 0, f"Return code {retcode}"
    # _unt() # initialize UnitRegistry (otherwise Unit cannot be used)
    # test_init(_unt())
    # test_auto_extreme()
    test_range_spec()
