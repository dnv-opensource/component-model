"""Test the pint package and identify the functions we need for this package"""

import logging

import pytest
from pint import UnitRegistry

from component_model.utils.logger import get_module_logger

logger = get_module_logger(__name__, level=logging.INFO)

_reg = UnitRegistry(system="SI", autoconvert_offset_to_baseunit=True)  # , auto_reduce_dimensions=True)


def test_needed_functions():
    _reg = UnitRegistry(system="SI", autoconvert_offset_to_baseunit=True)  # , auto_reduce_dimensions=True)
    logger.info(f"AVAILABLE UNITS: {dir(_reg.sys.SI)}")
    logger.info(
        f"degrees_Celsius defined? {'degree_Celsius' in _reg} but not degrees_Celsius: {'degrees_Celsius' in _reg}"
    )
    logger.info(f"Unit System {_reg.default_system}")
    logger.info(f"Implicit and explicit def {0.2 * _reg.kg} {_reg.Quantity(0.2, 'kg')}")
    logger.info(f"String parsing: {_reg('0.2kg')} {_reg('0.2 kg')} {_reg('kg')} {_reg('1.3e5')}")
    logger.info(f"Base Units {_reg.Quantity(1.0, 'ft').to_base_units()} {_reg('0.2 kg').to_base_units()}")
    logger.info(
        f"Disect {_reg.Quantity(1.0, 'ft').magnitude} {_reg.Quantity(1.0, 'ft').units} {_reg.Quantity(1.0, 'ft').dimensionality}"
    )
    logger.info(
        f"Temperature {_reg.Quantity(38.0, 'degC')} {_reg('38.0*degK')} {_reg.Quantity(38.0, 'degF')} {_reg.Quantity(38.0, 'degF').to_base_units()}"
    )  # string recognition works only for Kelvin, because the others have an offset
    q = _reg("38 degC")
    logger.info(f"Temperature from string {q} {q.magnitude} {q.units} {q.dimensionality} {q.to_base_units()}")
    #    u0 = str(q.units)
    qB = q.to_base_units()
    val = qB.magnitude
    uB = str(qB.units)
    qInv = _reg.Quantity(val, uB)
    logger.info(f"QINV {type(qInv)}")
    assert q == qInv.to("degC")
    q = _reg("36 deg")  # angle
    assert str(q.to_base_units().units) == "radian"
    assert str(q.units) == "degree"
    assert str(_reg("1.0%").to_base_units().units) == "dimensionless"
    q = _reg("3.6 N")
    logger.info("dimensionality of N: " + "".join(f"{x}:{q.dimensionality[x]!s}, " for x in q.dimensionality))
    assert q.dimensionality["[time]"] == -2
    assert q.units == "newton"
    logger.info(q.to_base_units().units)
    logger.info(q.to_reduced_units())
    assert q.check({"[mass]": 1, "[length]": 1, "[time]": -2})
    assert q.check("[force]"), "It is also a force (derived dimension)"
    q = _reg("2 m^2")
    # logger.info("COMPATIBLE UNITS: ", _reg.get_compatible_units(q ))
    assert q.is_compatible_with("square_mile"), "Check compatibility with a given unit"
    assert q.check("[area]"), "Check compatibility with derived dimensions"
    q = _reg("1 mol")
    logger.info(f"SUBSTANCE: {q} {q.to_base_units()} {q.dimensionality}")
    q = _reg("100 cd")
    logger.info(
        f"LUMINOSITY: {q} {q.to_base_units()} {q.dimensionality} {_reg.get_base_units(next(iter(_reg.get_compatible_units(q.dimensionality))))}"
    )
    q = _reg("9 degrees/s")
    logger.info(f"RAD/s: {q.check('rad/s')} {q.check('s/rad')} {q} {q.to_base_units()} {q.dimensionality}")
    q = _reg("9 s/deg")
    logger.info(
        f"RAD/s: {q.check('rad/s')} {q.check(dimension={'s': 1, 'rad': -1})} {q} {q.to_base_units()} {q.dimensionality}"
    )
    logger.info(f"30 degrees in base units: {_reg('30 deg').to_base_units().magnitude} radians.")


# def test_split():
#     assert Unit("kg".split()) == (1.0, "kg")
#     assert Unit("0.2kg".split()) == (0.2, "kg")
#     assert Unit.split("0.2 kg") == (0.2, "kg")
#     assert Unit.split("0.2E-4 m") == (2e-5, "m")
#     assert Unit.split(0.2e-4) == (2e-5, "")
#     assert tuple([Unit.split(u) for u in ("3m", "45 deg", 0)]) == ((3, "m"), (45, "deg"), (0, ""))
#
#
# def test_quantity():
#     assert Unit.quantity("30%") == (0.3, "%", "percent")
#     assert Unit.quantity("m") == (1.0, "m", "length")
#     assert Unit.quantity("-10 deg") == (-0.17453292519943295, "deg", "angle")
#     with pytest.raises(Exception) as err:
#         Unit.quantity("0.2um")  # undefined unit
#     logger.info(f"ERROR {err}")
#     assert Unit.quantity("0.2kg", "mass") == (0.2, "kg", "mass")
#     with pytest.raises(Exception) as err:
#         Unit.quantity("0.2kg", "length")  # unit not as expected
#     logger.info(f"ERROR {err}")
#
#
# def test_convert():
#     assert abs(Unit.convert(1.0, "nm") - 0.0005399568034557236) < 1e-7
#     assert abs(Unit.convert("3kg", "lb") - 6.613867865546327) < 1e-7
#
#
# def test_get_standard_unit():
#     assert Unit.get_standard_unit("mass") == "kg"

if __name__ == "__main__":
    retcode = pytest.main(args=["-rA", "-v", __file__])
    assert retcode == 0, f"Non-zero return code {retcode}"
    # test_needed_functions()
