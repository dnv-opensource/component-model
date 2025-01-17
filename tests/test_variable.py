import logging
import math
import xml.etree.ElementTree as ET  # noqa: N817
from enum import Enum

import numpy as np
import pytest
from pythonfmu.enums import Fmi2Causality as Causality  # type: ignore
from pythonfmu.enums import Fmi2Variability as Variability  # type: ignore

from component_model.caus_var_ini import Initial
from component_model.model import Model  # type: ignore
from component_model.utils.logger import get_module_logger  # type: ignore
from component_model.variable import (  # type: ignore
    Check,
    Variable,
    VariableInitError,
    VariableRangeError,
    cartesian_to_spherical,
    spherical_to_cartesian,
)

logger = get_module_logger(__name__, level=logging.INFO)

class DummyModel(Model):
    def __init__(self, name, **kwargs):
        super().__init__(name=name, description="Just a dummy model to be able to do testing", **kwargs)
    def do_step(self, time:int|float, dt:int|float):
        return True

def arrays_equal(arr1, arr2, dtype="float", eps=1e-7):
    assert len(arr1) == len(arr2), "Length not equal!"

    for i in range(len(arr1)):
        # assert type(arr1[i]) == type(arr2[i]), f"Array element {i} type {type(arr1[i])} != {type(arr2[i])}"
        assert abs(arr1[i] - arr2[i]) < eps, f"Component {i}: {arr1[i]} != {arr2[i]}"


def tuples_nearly_equal(tuple1: tuple, tuple2: tuple, eps=1e-10):
    """Check whether the values in tuples (of any tuple-structure) are nearly equal"""
    assert isinstance(tuple1, tuple), f"{tuple1} is not a tuple"
    assert isinstance(tuple2, tuple), f"{tuple2} is not a tuple"
    assert len(tuple1) == len(tuple2), f"Lenghts of tuples {tuple1}, {tuple2} not equal"
    for t1, t2 in zip(tuple1, tuple2, strict=False):
        if isinstance(t1, tuple):
            assert isinstance(t2, tuple), f"Tuple expected. Found {t2}"
            assert tuples_nearly_equal(t1, t2)
        elif isinstance(t1, float) or isinstance(t2, float):
            assert t1 == t2 or abs(t1 - t2) < eps, f"abs({t1} - {t2}) >= {eps}"
        else:
            assert t1 == t2, f"{t1} != {t2}"
    return True


def test_var_check():
    ck = Check.u_none | Check.r_none
    assert Check.r_none in ck, "'Check.u_none | Check.r_none' sets both unit and range checking to None"
    assert Check.u_none in ck, "'Check.u_none | Check.r_none' sets both unit and range checking to None"
    assert (Check.u_none | Check.r_none) == Check.none, "'none' combines 'u_none' and 'r_none'"
    ck = Check.u_all | Check.r_check
    assert Check.r_check in ck
    assert Check.u_all in ck
    assert ck & Check.units | Check.u_all, "filter the combined flag on unit"
    assert ck & Check.ranges | Check.r_check, "filter the combined flag on range"
    ck = Check.r_check  # check only range
    assert Check.r_check in ck
    assert Check.u_all not in ck
    assert not Check.none & ck
    ck = Check.u_all  # check only units
    assert Check.u_all in ck
    assert Check.r_check not in ck
    assert not Check.none & ck
    ck = ck & Check.units
    assert ck == Check.units, "Switched off range checking"
    ck = Check.ranges
    ck = ck & Check.units
    assert ck == Check.none, f"Switched off range checking: {ck}. None left"


def test_auto_type():
    assert Variable.auto_type(1) is float, "int not allowed (default)"
    assert Variable.auto_type(1, allow_int=True) is int, "int allowed"
    assert Variable.auto_type(0.99, allow_int=True) is float
    assert Variable.auto_type(0.99, allow_int=False) is float
    assert Variable.auto_type((1, 2, 0.99), allow_int=False) is float
    assert Variable.auto_type((1, 2, 0.99), allow_int=True) is float, "Ok by our rules"
    assert Variable.auto_type((1, 2, 3), allow_int=True) is int
    assert Variable.auto_type((True, False, 3), allow_int=True) is int
    assert Variable.auto_type((True, False), allow_int=True) is bool
    assert Variable.auto_type((True, False), allow_int=False) is bool
    assert Variable.auto_type((True, 1, 9.9), allow_int=False) is bool
    #     with pytest.raises(VariableInitError) as err: # that goes too far
    #         assert Variable.auto_type( (True,1, 9.9), allow_int=False) == float
    #     assert str(err.value).startswith("Incompatible variable types")


def test_spherical_cartesian():
    for vec in [
        (0, 0, 0),
        (0, 0, 1),
        (0, 1, 0),
        (0, 1, 1),
        (1, 0, 0),
        (1, 0, 1),
        (1, 1, 0),
        (1, 1, 1),
    ]:
        sVec = cartesian_to_spherical(vec)
        _vec = spherical_to_cartesian(sVec)
        arrays_equal(np.array(vec, dtype="float"), _vec)


def init_model_variables():
    """Define model and a few variables for various tests"""
    mod = DummyModel("MyModel")
    assert mod.name == "MyModel"
    assert mod.ureg is not None

    myInt = Variable(
        mod,
        "myInt",
        description="A integer variable",
        causality="parameter",
        variability="fixed",
        start=99,
        rng=(0, 100),
        typ=int,
        annotations=None,
        value_check=Check.all,
    )
    myInt2 = Variable(
        mod,
        "myInt2",
        description="A integer variable without range checking",
        value_reference=99,  # manual valueReference
        causality="input",
        variability="continuous",
        start=99,
        rng=(0, 100),
        typ=int,
        value_check=Check.none,
    )
    myFloat = mod.add_variable(
        "myFloat",
        description="A float variable",
        causality="input",
        variability="continuous",
        start="99.0%",
        rng=(0.0, None),
    )
    myEnum = Variable(
        mod,
        "myEnum",
        description="An enumeration variable",
        causality="output",
        variability="discrete",
        start=Causality.parameter,
    )
    myBool = Variable(
        mod,
        "myBool",
        description="A boolean variable",
        causality="parameter",
        variability="fixed",
        start=True,
    )
    myStr = Variable(
        mod,
        "myStr",
        description="A string variable",
        typ=str,
        causality="parameter",
        variability="fixed",
        start="Hello World!",
    )
    myNP = Variable(
        mod,
        "myNP",
        description="A NP variable",
        causality="parameter",
        variability="fixed",
        start=("1.0m", "2deg", "3rad"),
        rng=((0, "3m"), ("1 deg", "5 deg"), (float("-inf"), "5rad")),
    )
    myNP2 = Variable(
        mod,
        "myNP2",
        description="A NP variable with on_set and on_step",
        causality="input",
        variability="continuous",
        start=("1.0", "2.0", "3.0"),
        rng=((0, float("inf")), (0, float("inf")), (0, float("inf"))),
        on_set=lambda val: 0.9 * val,
        on_step=lambda t, dt: mod.myNP2[0](dt * mod.myNP2[0]),
    )
    return (mod, myInt, myInt2, myFloat, myEnum, myStr, myNP, myNP2, myBool)


def test_init():
    mod = DummyModel("MyModel")
    (
        mod,
        myInt,
        myInt2,
        myFloat,
        myEnum,
        myStr,
        myNP,
        myNP2,
        myBool,
    ) = init_model_variables()
    assert myInt.typ is int
    assert myInt.description == "A integer variable"
    assert myInt.causality == Causality.parameter
    assert myInt.variability == Variability.fixed
    assert myInt.initial == Initial.exact
    assert myInt.check == Check.all
    # internally packed into tuple:
    assert myInt.start == (99,)
    assert myInt.range == ((0, 100),)
    assert myInt.unit == ("dimensionless",)
    assert myInt.display == (None,)
    assert myInt.check_range(50)
    assert not myInt.check_range(110)
    assert mod.myInt == 99, "Value directly accessible as model variable"
    mod.myInt = 110
    assert mod.myInt == 110, "Internal changes not range-checked!"
    with pytest.raises(VariableRangeError) as err:  # ... but getter() detects the range error
        _ = myInt.getter()
    assert str(err.value) == "getter(): Value 110 outside range."
    assert mod.myInt == 110, f"Value {mod.myInt} should still be unchanged"
    myInt.setter(50)
    assert mod.myInt == 50, f"Value {mod.myInt} changed back."
    mod.set_integer(((mod.variable_by_name("myInt").value_reference),), (99,))  # simulate setting from outside
    assert mod.get_integer(((mod.variable_by_name("myInt").value_reference),)) == [99]

    assert myFloat.typ is float
    assert myFloat.causality == Causality.input
    assert myFloat.variability == Variability.continuous
    assert myFloat.initial == Initial.none, f"initial: {myFloat.initial}"
    assert myFloat.check == Check.all
    # internally packed into tuple:
    assert myFloat.start == (0.99,)
    assert myFloat.range == ((0, 99.0),), f"Range: {myFloat.range} in display units."
    assert myFloat.unit == ("dimensionless",)
    assert myFloat.display[0][0] == "percent", f"Display: {myFloat.display[0][0]}"
    assert myFloat.display[0][1](99) == 0.99, "Transform from dimensionless to percent"
    assert myFloat.display[0][2](0.99) == 99, "... and back."
    assert myFloat.check_range(0.5)
    assert not myFloat.check_range(1.0, disp=False), "Check as internal units"
    assert not myFloat.check_range(100.0), "Check as display units"
    assert mod.myFloat == 0.99, "Value directly accessible as model variable"
    mod.myFloat = 1.0
    assert mod.myFloat == 1.0, "Internal changes not range-checked!"
    with pytest.raises(VariableRangeError) as err:  # ... but getter() detects the range error
        _ = myFloat.getter()
    assert str(err.value) == "getter(): Value 100.0 outside range."
    assert mod.myFloat == 1.0, f"Value {mod.myFloat} should still be unchanged"
    myFloat.setter(50)
    assert mod.myFloat == 0.5, f"Value {mod.myFloat} changed back."
    mod.set_real(((mod.variable_by_name("myFloat").value_reference),), (99,))  # simulate setting from outside
    assert mod.get_real(((mod.variable_by_name("myFloat").value_reference),)) == [99]

    assert isinstance(myEnum.start[0], Enum), f"Type: {myEnum.typ}"
    assert issubclass(myEnum.typ, Enum), "Enums are always derived"
    assert myEnum.causality == Causality.output
    assert myEnum.variability == Variability.discrete
    assert myEnum.initial == Initial.calculated, f"initial: {myEnum.initial}"
    assert myEnum.check == Check.all
    # internally packed into tuple:
    assert myEnum.start == (Causality.parameter,)
    assert myEnum.range == ((0, 4),), f"Range: {myEnum.range}"
    assert myEnum.unit == ("dimensionless",)
    assert myEnum.display[0] is None, f"Display: {myEnum.display[0]}"
    assert myEnum.check_range(1)
    assert not myEnum.check_range(7)
    assert mod.myEnum == Causality.parameter, f"Value {mod.myEnum} directly accessible as model variable"
    mod.myEnum = Causality.input
    assert mod.myEnum == Causality.input, "Not possible to supply a wrong value with the right type!"
    myEnum.setter(Causality.local)
    assert myEnum.getter() == 4, f"Value {myEnum.getter()}. Translated to int!"
    mod.set_integer(((mod.variable_by_name("myEnum").value_reference),), (2,))  # simulate setting from outside
    assert mod.get_integer(((mod.variable_by_name("myEnum").value_reference),)) == [2]

    assert myBool.typ is bool
    assert myBool.causality == Causality.parameter
    assert myBool.variability == Variability.fixed
    assert myBool.initial == Initial.exact
    assert myBool.check == Check.all
    # internally packed into tuple:
    assert myBool.start == (True,)
    assert myBool.range == ((False, True),)
    assert myBool.unit == ("dimensionless",)
    assert myBool.display == (None,)
    assert myBool.check_range(True)
    assert myBool.check_range(100.5), "Any number will work"
    assert not myBool.check_range("Hei"), "But non-numbers are rejected"
    assert mod.myBool, "Value directly accessible as model variable"
    mod.myBool = 100
    assert mod.myBool == 100, "Internal changes not range-checked!"
    assert myBool.getter(), "Converted in getter()"
    myBool.setter(False)
    assert not mod.myBool, f"Value {mod.myBool} changed."
    mod.set_boolean(((mod.variable_by_name("myBool").value_reference),), (True,))  # simulate setting from outside
    assert mod.get_boolean(((mod.variable_by_name("myBool").value_reference),)) == [True]

    assert myStr.typ is str
    assert myStr.causality == Causality.parameter
    assert myStr.variability == Variability.fixed
    assert myStr.initial == Initial.exact, f"initial: {myStr.initial}"
    assert myStr.check == Check.all
    # internally packed into tuple:
    assert myStr.start == ("Hello World!",)
    assert myStr.range == (("", ""),), f"Range: {myStr.range}. Basically irrelevant"
    assert myStr.unit == ("dimensionless",), f"Unit {myStr.unit}"
    assert myStr.display[0] is None, f"Display: {myStr.display[0]}"
    assert myStr.check_range(0.5), "Everything is ok"
    assert mod.myStr == "Hello World!", f"Value {mod.myStr} directly accessible as model variable"
    mod.myStr = 1.0
    assert mod.myStr == 1.0, f"Not converted to str when internally set: {mod.myStr}"
    assert isinstance(myStr.getter(), str), "getter() converts to str"
    myStr.setter("Hei")
    assert mod.myStr == "Hei", f"New value {mod.myStr}."
    mod.set_string(((mod.variable_by_name("myStr").value_reference),), ("Hello",))  # simulate setting from outside
    assert mod.get_string(((mod.variable_by_name("myStr").value_reference),)) == ["Hello"]

    assert myNP.typ is float
    assert myNP == mod.variable_by_name("myNP")
    assert myNP.description == "A NP variable"
    assert mod.variable_by_name("myNP[1]") == mod.variable_by_name("myNP"), "Returns always the parent"
    assert myNP.causality == Causality.parameter
    assert myNP.variability == Variability.fixed
    assert myNP.initial == Initial.exact
    assert myNP.check == Check.all
    # internally packed into tuple:
    assert myNP.start == (1, math.radians(2), 3)
    tuples_nearly_equal(myNP.range, ((0, 3), (1, 5), (float("-inf"), 5)))
    assert not myNP.check_range(5.1, idx=1), "Checks performed on display units!"
    assert not myNP.check_range(0.9, idx=1), "Checks performed on display units!"
    assert myNP.unit == ("meter", "radian", "radian"), f"Units: {myNP.unit}"
    assert myNP.display[0] is None
    assert myNP.display[1][0] == "degree"
    assert myNP.display[2] is None
    assert myNP.check_range((2, 3.5, 4.5))
    assert not myNP.check_range((2, 3.5, 6.3))
    assert mod.myNP[1] == math.radians(2), "Value directly accessible as model variable"
    mod.myNP[1] = -1.0
    assert mod.myNP[1] == -1.0, "Internal changes not range-checked!"
    with pytest.raises(VariableRangeError) as err:  # ... but getter() detects the range error
        _ = myNP.getter()
    #       assert str(err.value) == "getter(): Value [1.0, -57.29577951308233, 3.0] outside range."
    assert mod.myNP[1] == -1.0, f"Value {mod.myNP} should still be unchanged"
    mod.myNP = np.array((1.5, 2.5, 3.5), float)
    assert np.linalg.norm(mod.myNP) == math.sqrt(1.5**2 + 2.5**2 + 3.5**2), "np calculations are done on value"
    myNP.setter((1.0, 1.0, 1.0))
    arrays_equal(mod.myNP, (1.0, math.radians(1.0), 1.0))
    arrays_equal(myNP.getter(), (1.0, 1.0, 1.0))  # getter shows display units
    vr0 = mod.variable_by_name("myNP").value_reference
    mod.set_real((vr0, vr0 + 1, vr0 + 2), (2.0, 2.0, 2.0))  # simulate setting from outside
    arrays_equal(mod.get_real((vr0, vr0 + 1, vr0 + 2)), [2.0, 2.0, 2.0])
    arrays_equal(mod.get_real((vr0, vr0 + 1, vr0 + 2)), [2.0, 2.0, 2.0])  # array not changed by getter (need copy)

    with pytest.raises(Exception) as err:
        _ = Variable(
            mod,
            "myInt",
            description="An integer variable with a non-unique name",
            causality="input",
            variability="continuous",
            typ=int,
            start="99.9%",
            rng=(0, "100%"),
            annotations=None,
        )
    assert str(err.value).startswith("Variable name myInt already used as index 0 in model MyModel")

    with pytest.raises(VariableInitError) as err:
        myInt = Variable(
            mod,
            "myBool",
            description="A second integer variable with erroneous range",
            causality="parameter",
            variability="fixed",
            start="99",
            rng=(),
            annotations=None,
            typ=int,
        )
    assert str(err.value).startswith("Range must be specified for int variable")
    assert myFloat.range[0][1] == 99.0
    assert myEnum.range[0] == (0, 4)
    assert myEnum.check_range(Causality.parameter)
    assert myStr.range == (("", ""),), "Just a placeholder. Range of str is not checked"
    assert myBool.typ is bool


def test_range():
    """Test the various ways of providing a range for a variable"""
    mod = DummyModel("MyModel2", instance_name="MyModel2")
    with pytest.raises(VariableInitError) as err:
        int1 = Variable(mod, "int1", start=1)
    assert (
        str(err.value)
        == "Range must be specified for int variable <class 'component_model.variable.Variable'> or use float."
    )
    int1 = Variable(mod, "int1", start=1, rng=(0, 5))  # that works
    mod.int1 = 6
    with pytest.raises(VariableRangeError) as err:  # causes an error
        _ = int1.getter()
    assert str(err.value) == "getter(): Value 6.0 outside range."
    float1 = Variable(mod, "float1", start=1, typ=float)  # explicit type
    assert float1.range == ((float("-inf"), float("inf")),), "Auto_extreme. Same as rng=()"
    float2 = Variable(mod, "float2", start=1.0, rng=None)  # implicit type through start value and no range
    assert float2.range == ((1.0, 1.0),), "No range."
    with pytest.raises(VariableRangeError) as err:
        float2.setter(2.0)

    np1 = Variable(mod, "np1", start=("1.0m", 2, 3), rng=((0, "3m"), None, tuple()))
    assert np1.range == ((0.0, 3.0), (2.0, 2.0), (float("-inf"), float("inf")))


def test_dirty():
    """Test the dirty mechanism"""
    mod = DummyModel("MyModel2", instance_name="MyModel2")
    myNP = Variable(
        mod,
        "myNP",
        description="A NP variable",
        causality="parameter",
        variability="fixed",
        start=("1.0m", "2deg", "3rad"),
        on_set=lambda x: 0.5 * x,
        rng=((0, "3m"), (0, float("inf")), (float("-inf"), "5rad")),
    )
    assert myNP.typ is float, f"Type {myNP.typ}"
    myNP.setter(np.array((2, 1, 4), float))
    assert myNP not in mod.dirty, "Not dirty, because the whole variable was changed"
    arrays_equal(mod.myNP, [0.5 * 2.0, 0.5 * math.radians(1), 0.5 * 4])  # ... and on_set has been run
    mod.set_real([1], [9.9])
    assert myNP in mod.dirty, "Dirty. on_set has not been run."
    arrays_equal(myNP.getter(), [1, 9.9, 2])  # on_set not yet run
    mod.dirty_do()
    arrays_equal(myNP.getter(), [0.5 * 1, 0.5 * 9.9, 0.5 * 2])  # on_set run


def test_var_ref():
    Model.instances = []  # reset
    (
        mod,
        myInt,
        myInt2,
        myFloat,
        myEnum,
        myStr,
        myNP,
        myNP2,
        myBool,
    ) = init_model_variables()
    # print(mod.vars)
    assert mod.vars[99].name == "myInt2"
    assert mod.vars[6].name == "myNP"
    assert mod.vars[7] is None, "a sub-element"
    var, sub = mod.ref_to_var(7)
    assert var.name == "myNP"
    assert sub == 1
    assert mod.variable_by_name("myInt2").name == "myInt2"
    assert mod.variable_by_name("myInt2").value_reference == 99
    # mod.variable_by_value(mod.myInt) deleted


def test_vars_iter():
    Model.instances = []  # reset
    (
        mod,
        myInt,
        myInt2,
        myFloat,
        myEnum,
        myStr,
        myNP,
        myNP2,
        myBool,
    ) = init_model_variables()
    assert list(mod.vars_iter(float)) == [myFloat, myNP, myNP2]
    assert list(mod.vars_iter(float))[0].name == "myFloat"
    assert list(mod.vars_iter(float))[1].name == "myNP"
    assert list(mod.vars_iter(key=Variability.discrete))[0].name == "myEnum"
    assert list(mod.vars_iter(key=Causality.input))[1].name == "myFloat"
    assert (
        list(mod.vars_iter(key=lambda x: x.causality == Causality.input or x.causality == Causality.output))[2].name
        == "myEnum"
    )


def test_get():
    Model.instances = []  # reset
    (
        mod,
        myInt,
        myInt2,
        myFloat,
        myEnum,
        myStr,
        myNP,
        myNP2,
        myBool,
    ) = init_model_variables()
    # print( "".join( str(i)+":"+mod.vars[i].name+", " for i in range( len(mod.vars)) if mod.vars[i] is not None))
    assert mod._get([0, 99], int) == [99, 99]
    assert mod.get_integer([0, 99]) == [99, 99]
    assert mod.get_integer([0, 99]) == [99, 99]
    with pytest.raises(AssertionError) as err:
        _ = mod.get_real([0, 99])
    assert str(err.value).startswith("Invalid type in 'get_")
    assert mod.get_real([2]) == [99.0], f"Got value {mod.get_real([2])} (converted to %)"
    assert mod.get_string([5])[0] == "Hello World!"
    assert abs(mod.get_real([7])[0] - 2.0) < 1e-14, "Second element of compound variable"
    assert mod.get_real([6])[0] == 1.0
    assert mod.vars[6].name == "myNP"
    var, sub = mod.ref_to_var(7)
    assert var.name == "myNP" and sub == 1, "Second element of NP variable"
    assert len(var) == 3
    assert mod.variable_by_name("myNP").value_reference == 6
    arrays_equal(mod.get_real([6, 7, 8]), [1.0, 2.0, 3.0])  # translated back to degrees
    with pytest.raises(AssertionError) as err:
        _ = mod.get_real([9, 12])
    assert str(err.value) == "Variable with valueReference=12 does not exist in model MyModel"


def test_set():
    Model.instances = []  # reset
    (
        mod,
        myInt,
        myInt2,
        myFloat,
        myEnum,
        myStr,
        myNP,
        myNP2,
        myBool,
    ) = init_model_variables()
    # print( "".join( str(i)+":"+mod.vars[i].name+", " for i in mod.vars if mod.vars[i] is not None))
    mod.set_integer([0, 99], [60, 61])
    assert mod.vars[0].getter() == 60
    assert mod.myInt == 60
    assert mod.vars[99].getter() == 61
    with pytest.raises(AssertionError) as err:
        mod.set_integer([6, 7], [2.0, "30 deg"])
    assert str(err.value) == "Invalid type in 'set_<class 'int'>'. Found variable myNP with type <class 'float'>"
    mod.set_real([6, 7], [2.0, 3.0])  # "3 deg"])


# @pytest.mark.skip()
def test_xml():
    Model.instances = []  # reset
    mod = DummyModel("MyModel")
    myNP2 = Variable(
        mod,
        "Test9",
        description="A NP variable ...",
        causality="input",
        variability="continuous",
        start=("1m", "2deg", "3 deg"),
        rng=((0, "3m"), None, None),
    )
    lst = myNP2.xml_scalarvariables()
    assert len(lst) == 3
    expected = b'<ScalarVariable name="Test9[0]" valueReference="0" description="A NP variable ..." causality="input" variability="continuous"><Real start="1.0" min="0.0" max="3.0" unit="meter" /></ScalarVariable>'
    assert ET.tostring(lst[0]) == expected, f"{ET.tostring(lst[0])}"
    expected = b'<ScalarVariable name="Test9[1]" valueReference="1" description="A NP variable ..." causality="input" variability="continuous"><Real start="0.03490658503988659" min="1.9999999999999993" max="2.0000000000000013" unit="radian" displayUnit="degree" /></ScalarVariable>'
    assert ET.tostring(lst[1]) == expected, f"{ET.tostring(lst[1])}"
    expected = b'<ScalarVariable name="Test9[2]" valueReference="2" description="A NP variable ..." causality="input" variability="continuous"><Real start="0.05235987755982989" min="2.9999999999999996" max="3.0000000000000013" unit="radian" displayUnit="degree" /></ScalarVariable>'
    assert ET.tostring(lst[2]) == expected, f"{ET.tostring(lst[2])}"

    myInt = Variable(
        mod,
        "myInt",
        description="A integer variable",
        causality="parameter",
        variability="fixed",
        start="99%",
        rng=(0, "100%"),
        annotations=None,
        value_check=Check.all,
    )
    lst = myInt.xml_scalarvariables()
    expected = b'<ScalarVariable name="myInt" valueReference="3" description="A integer variable" causality="parameter" variability="fixed" initial="exact"><Real start="0.99" min="0.0" max="100.0" unit="dimensionless" displayUnit="percent" /></ScalarVariable>'
    assert ET.tostring(lst[0]) == expected, ET.tostring(lst[0])


def test_on_set():
    (
        mod,
        myInt,
        myInt2,
        myFloat,
        myEnum,
        myStr,
        myNP,
        myNP2,
        myBool,
    ) = init_model_variables()
    # print( "".join( str(i)+":"+mod.vars[i].name+", " for i in range( len(mod.vars)) if mod.vars[i] is not None))
    arrays_equal(mod.myNP2, (1, 2, 3))
    myNP2.setter(np.array((4, 5, 6), float))
    arrays_equal(mod.myNP2, (0.9 * 4, 0.9 * 5, 0.9 * 6))  # on_set run, because whole array is set
    mod.set_real([10, 11], [7, 8])
    arrays_equal(mod.myNP2, (0.9 * 4, 7, 8))
    mod.dirty_do()
    arrays_equal(mod.myNP2, (0.9 * 0.9 * 4, 0.9 * 7, 0.9 * 8))


if __name__ == "__main__":
    retcode = pytest.main(["-rP -s -v", __file__])
    assert retcode == 0, f"Return code {retcode}"
    # test_range()
    # test_var_check()
    # test_spherical_cartesian()
    # test_auto_type()
    # test_init()
    # test_dirty()
    # test_var_ref()
    # test_vars_iter()
    # test_get()
    # test_set()
    # test_on_set()
    # test_xml()
