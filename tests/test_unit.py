from math import degrees, radians
from typing import Any

import pytest
from pint import UnitRegistry

from component_model.unit import Unit


@pytest.fixture
def ureg(scope: str = "module", autouse: bool = True):
    return _ureg()


def _ureg():
    _registry = Unit.ensure_unit_registry("SI")
    assert isinstance(_registry, UnitRegistry)
    return _registry


def test_parsing(ureg: UnitRegistry[Any]):
    u1 = Unit()
    # default values:
    assert u1.u == "dimensionless"
    assert u1.du is None
    val = u1.parse_quantity("9.9m")
    assert val == 9.9
    assert u1.u == "meter"
    assert u1.du is None
    val = u1.parse_quantity("9.9inch")
    assert u1.to_base is not None and u1.from_base is not None
    assert val == u1.to_base(9.9), f"Found val={val}"
    assert u1.u == "meter"
    assert u1.du == "inch"
    assert abs(123.456 - u1.to_base(u1.from_base(123.456))) < 1e-13, f"Found {u1.to_base(u1.from_base(123.456))}"
    val = u1.parse_quantity("99.0%")
    assert val == 0.99
    assert u1.u == "dimensionless"
    assert u1.du == "percent"
    assert str(u1) == "Unit dimensionless, display:percent. Offset:0.0, factor:0.01"
    # Note: the following works only if autoconvert_offset_to_baseunit=True within UnitRegistry
    uf = Unit("0.0 degF")  # possible to initialize as degF, but base-value is not returned
    assert uf.u == "kelvin"
    assert uf.du == "degree_Fahrenheit"
    assert uf.parse_quantity("0.0 degF") == 255.37222222222223
    assert uf.to_base is not None and uf.to_base(0.0) == 255.37222222222223


def test_make(ureg: UnitRegistry[Any]):
    val, unit = Unit.make("2m")
    assert val[0] == 2
    assert unit[0].u == "meter", f"Found {unit[0].u}"
    assert unit[0].du is None
    val, unit = Unit.make("Hello World", typ=str)
    assert val[0] == "Hello World"
    assert unit[0].u == "dimensionless"
    assert unit[0].du is None
    val, unit = Unit.make("99.0%")
    assert val[0] == 0.99
    assert unit[0].u == "dimensionless"
    assert unit[0].du == "percent"


def test_make_tuple(ureg: UnitRegistry[Any]):
    vals, units = Unit.make_tuple(("2m", "3deg", "0.0 degF"))
    k2degc = 273.15
    assert units[0].u == "meter"
    assert units[0].du is None
    assert vals[0] == 2
    assert units[1].u == "radian", f"Found {units[1].u}"
    assert units[1].du == "degree"
    assert units[1].to_base is not None
    assert units[1].to_base(1.0) == radians(1.0)
    assert units[1].from_base is not None
    assert units[1].from_base(1.0) == degrees(1.0)
    assert vals[1] == radians(3)
    assert units[2].u == "kelvin", f"Found {units[2].u}"
    assert units[2].du == "degree_Fahrenheit", f"Found {units[2].du}"
    assert units[2].from_base is not None
    assert abs(units[2].from_base(k2degc) - (k2degc * 9 / 5 - 459.67)) < 1e-10
    assert units[2].to_base is not None
    assert abs(units[2].to_base(0.0) - (0.0 + 459.67) * 5 / 9) < 1e-10, (
        f"Found {units[2].to_base(0.0)}, {(0.0 + 459.67) * 5 / 9}"
    )


def test_derivative(ureg: UnitRegistry[Any]):
    bv, bu = Unit.make_tuple(("2m", "3deg"))
    vals, units = Unit.derivative(bu)
    assert vals == (0.0, 0.0)
    assert units[0].u == "meter/s"
    assert units[0].du is None
    assert units[1].u == "radian/s", f"Found {units[1].u}"
    assert units[1].du == "degree/s"
    assert units[1].to_base == bu[1].to_base
    assert units[1].from_base == bu[1].from_base


def test_compatible(ureg: UnitRegistry[Any]):
    v, u = Unit.make_tuple(("2m", "3deg"))
    ck, q = u[0].compatible("4m", strict=True)
    assert ck
    assert q == 4
    ck, q = u[1].compatible("5 radian", strict=True)
    assert not ck, "Not compatible for 'strict'"
    ck, q = u[1].compatible("5 radian", strict=False)
    assert ck, "Ok for non-strict"
    ck, q = u[0].compatible("5 radian", strict=False)
    assert not ck, "Totally wrong units"


if __name__ == "__main__":
    retcode = pytest.main(["-rP -s -v", __file__])
    assert retcode == 0, f"Return code {retcode}"
    # ureg = _ureg()
    # test_parsing(ureg)
    # test_make(ureg)
    # test_make_tuple(ureg)
    # test_derivative(ureg)
    # test_compatible(ureg)
