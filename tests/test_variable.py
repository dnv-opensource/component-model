import logging
import math

import numpy as np
import pytest
from component_model.logger import get_module_logger  # type: ignore
from component_model.model import Model  # type: ignore
from component_model.variable import (  # type: ignore
    VarCheck,
    Variable,
    VariableInitError,
    VariableNP,
    cartesian_to_spherical,
    spherical_to_cartesian,
)
from pythonfmu.enums import Fmi2Causality as Causality  # type: ignore
from pythonfmu.enums import Fmi2Variability as Variability  # type: ignore

logger = get_module_logger(__name__, level=logging.INFO)


def np_arrays_equal(arr1, arr2, dtype="float64", eps=1e-7):
    assert len(arr1) == len(arr2), "Length not equal!"
    if isinstance(arr2, (tuple, list)):
        arr2 = np.array(arr2, dtype=dtype)
    assert isinstance(arr1, np.ndarray) and isinstance(
        arr2, np.ndarray
    ), "At least one of the parameters is not an ndarray!"
    assert arr1.dtype == arr2.dtype, f"Arrays are of type {arr1.dtype} != {arr2.dtype}"

    for i in range(len(arr1)):
        assert abs(arr1[i] - arr2[i]) < eps, f"Component {i}: {arr1[i]} != {arr2[i]}"


def test_var_check():
    ck = VarCheck.u_none | VarCheck.r_none
    assert VarCheck.r_none in ck, "'VarCheck.u_none | VarCheck.r_none' sets both unit and range checking to None"
    assert VarCheck.u_none in ck, "'VarCheck.u_none | VarCheck.r_none' sets both unit and range checking to None"
    assert (VarCheck.u_none | VarCheck.r_none) == VarCheck.none, "'none' combines 'u_none' and 'r_none'"
    ck = VarCheck.u_all | VarCheck.r_check
    assert VarCheck.r_check in ck
    assert VarCheck.u_all in ck
    assert ck & VarCheck.units | VarCheck.u_all, "filter the combined flag on unit"
    assert ck & VarCheck.ranges | VarCheck.r_check, "filter the combined flag on range"
    ck = VarCheck.r_check  # check only range
    assert VarCheck.r_check in ck
    assert VarCheck.u_all not in ck
    assert not VarCheck.none & ck
    ck = VarCheck.u_all  # check only units
    assert VarCheck.u_all in ck
    assert VarCheck.r_check not in ck
    assert not VarCheck.none & ck


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
        np_arrays_equal(np.array(vec, dtype="float64"), _vec)


def init_model_variables():
    """Define model and a few variables for various tests"""
    mod = Model("MyModel")
    myInt = Variable(
        mod,
        "myInt",
        description="A integer variable",
        causality="parameter",
        variability="fixed",
        value0=99,
        rng=(0, 100),
        typ=int,
        annotations=None,
        value_check=VarCheck.all,
    )
    myInt2 = Variable(
        mod,
        "myInt2",
        description="A integer variable without range checking",
        causality="input",
        variability="continuous",
        value0=99,
        rng=(0, 100),
        typ=int,
        annotations=None,
        value_check=VarCheck.none,
    )
    myFloat = mod.add_variable(
        "myFloat",
        description="A float variable",
        causality="input",
        variability="continuous",
        value0="99.0%",
        rng=(0.0, None),
        annotations=None,
    )
    myEnum = Variable(
        mod,
        "myEnum",
        description="An enumeration variable",
        causality="output",
        variability="discrete",
        value0=Causality.parameter,
        annotations=None,
    )
    myStr = Variable(
        mod,
        "myStr",
        description="A string variable",
        typ=str,
        causality="parameter",
        variability="fixed",
        value0="Hello World!",
        annotations=None,
    )
    myNP = VariableNP(
        mod,
        "myNP",
        description="A NP variable",
        causality="parameter",
        variability="fixed",
        value0=("1.0m", "2deg", "3rad"),
        rng=((0, "3m"), (0, float("inf")), (float("-inf"), "5rad")),
    )
    myNP2 = VariableNP(
        mod,
        "myNP2",
        description="A NP variable with on_set and on_step",
        causality="input",
        variability="continuous",
        value0=("1.0", "2.0", "3.0"),
        rng=((0, float("inf")), (0, float("inf")), (0, float("inf"))),
        on_set=lambda val: (0.9 * val[0], 0.9 * val[1], 0.9 * val[2]),
        on_step=lambda t, dT: mod.myNP2[0](dT * mod.myNP2[0]),
    )
    myBool = Variable(
        mod,
        "myBool",
        description="A boolean variable",
        causality="parameter",
        variability="fixed",
        value0=True,
        annotations=None,
    )
    return (mod, myInt, myInt2, myFloat, myEnum, myStr, myNP, myNP2, myBool)


def test_init():
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
    # test _get_auto_extreme()
    assert Variable._get_auto_extreme(1.0) == (float("-inf"), float("inf"))
    with pytest.raises(VariableInitError) as err:
        print(Variable._get_auto_extreme(1))
    assert str(err.value).startswith(
        "Unlimited integer variables do not make sense in Python. Please provide explicit limits for variable "
    )
    assert myInt.range == (0, 100)
    assert myInt.name == "myInt"
    assert mod.myInt == 99, f"Found {mod.myInt}"
    mod.myInt = 50
    assert myInt.getter() == 50, f"Found {myInt}"
    mod.myInt = 101  # Out of range value {mod.myInt} accepted
    with pytest.raises(AssertionError) as err:
        myInt.setter(101)  # ... but not when set 'from outside'
    assert str(err.value).startswith("Range violation in variable myInt, value 101. Should be")
    mod.myInt2 = 101  # does not cause an error message
    with pytest.raises(Exception) as err:
        _ = Variable(
            mod,
            "myInt",
            description="An integer variable with a non-unique name",
            causality="input",
            variability="continuous",
            typ=int,
            value0="99.9%",
            rng=(0, "100%"),
            annotations=None,
        )
    assert str(err.value).startswith("Variable name myInt is not unique")

    with pytest.raises(VariableInitError) as err:
        myInt = Variable(
            mod,
            "myBool",
            description="A second integer variable with erroneous range",
            causality="parameter",
            variability="fixed",
            value0="99",
            rng=(),
            annotations=None,
            typ=int,
        )
    assert str(err.value).startswith(
        "Unlimited integer variables do not make sense in Python. Please provide explicit limits for variable "
    )

    # one example using add_variable on the model
    assert myFloat.range[1] == 0.99

    assert myEnum.range == (0, 5)
    assert myEnum.check_range(Causality.parameter)

    assert len(myStr.range) == 0
    assert myBool.type == bool


def test_variable_np():
    mod = Model("MyModel")
    myNP = VariableNP(
        mod,
        "myNP",
        description="A NP variable",
        causality="parameter",
        variability="fixed",
        value0=("1.0m", "2deg", "3rad"),
        rng=((0, "3m"), (0, float("inf")), (float("-inf"), "5rad")),
    )
    np_arrays_equal(myNP.value0, (1, math.radians(2), 3))
    assert myNP.name == "myNP"
    assert myNP.range[1] == (0, float("inf"))
    assert str(myNP.displayUnit[1][0]) == "degree"
    np_arrays_equal(np.array((1.0, math.radians(2.0), 3.0), dtype="float64"), mod.myNP)
    mod.myNP = np.array((1.5, 2.5, 3.5), dtype="float64")
    np_arrays_equal(mod.myNP, (1.5, 2.5, 3.5))
    assert np.linalg.norm(mod.myNP) == math.sqrt(1.5**2 + 2.5**2 + 3.5**2), "np calculations are done on value"
    mod.myNP[0] = 15  # does not cause an error (internal settings are not range-checked)
    with pytest.raises(AssertionError) as err:
        myNP.setter((4, 0, 4))
    assert str(err.value).startswith("Range violation in variable myNP[0]")
    with pytest.raises(AssertionError) as err:
        myNP.setter(None)  # check the new value and run .on_set if defined, causing an error
    assert str(err.value).startswith("Integer idx needed in this case")
    myNP2 = VariableNP(
        mod,
        "Test9",
        description="A NP variable with units included in initial values and partially fixed range",
        causality="input",
        variability="continuous",
        value0=("1m", "2deg", "3 deg"),
        rng=((0, "3m"), None, None),
    )
    assert list(myNP2.value0) == [1, math.radians(2), math.radians(3)]
    assert str(myNP2.unit[0]) == "meter"
    assert tuple(myNP2.range[0]) == (0, 3)
    assert tuple(myNP2.range[1]) == (math.radians(2), math.radians(2))

    myFloat2 = Variable(
        mod,
        "myNP2",
        description="A float variable with units included in initial value",
        causality="parameter",
        variability="fixed",
        value0="99%",
        rng=(0.0, None),
        annotations=None,
    )
    assert myFloat2.value0 == 0.99
    assert str(myFloat2.displayUnit[0]) == "percent"
    myFloat3 = Variable(
        mod,
        "myBool",
        description="A float variable with delta range",
        causality="parameter",
        variability="fixed",
        value0="99%",
        rng=None,
        annotations=None,
    )
    assert myFloat3.range == (0.99, 0.99)

    boom = VariableNP(
        mod,
        "test-boom",
        "The dimension and direction of the boom from anchor point to anchor point in m and spherical angles",
        causality="input",
        variability="continuous",
        value0=("5.0 m", "-90 deg", "0deg"),
        rng=(None, ("-180deg", "180deg"), None),
    )
    assert boom.value0[0] == 5.0, "Boom length"


def test_dirty():
    """Test the dirty mechanism"""
    mod = Model("MyModel2", instance_name="MyModel2")
    myNP = VariableNP(
        mod,
        "myNP",
        description="A NP variable",
        causality="parameter",
        variability="fixed",
        value0=("1.0m", "2deg", "3rad"),
        rng=((0, "3m"), (0, float("inf")), (float("-inf"), "5rad")),
    )
    assert myNP._type == np.float64, f"NP _type {myNP._type}"
    myNP.setter((2, 1, 4))
    assert list(myNP.getter()) == [1, math.radians(2), 3], "Vector should be unchanged"
    assert isinstance(mod.get_from_dirty(myNP), np.ndarray)
    assert list(mod.get_from_dirty(myNP)) == [2, 1, 4], f"Staged changed value: {mod.get_from_dirty(myNP)}"
    mod.dirty_do()
    assert list(myNP.getter()) == [2, 1, 4], f"Changed value: {myNP.getter()}"
    myNP.setter(2.5, idx=0)
    myNP.on_set(mod.get_from_dirty(myNP))
    assert list(myNP.getter()) == [2.5, 1, 4], "Vector changed"


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
    assert mod.vars[1].name == "myInt2"
    assert mod.vars[6] is None, "a sub-element"
    var, sub = mod.ref_to_var(6)
    assert var.name == "myNP"
    assert sub == 1
    assert mod.variable_by_name("myInt2").name == "myInt2"
    assert mod.variable_by_name("myInt2").value_reference == 1
    mod.variable_by_value(mod.myInt)


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
    assert mod._get([0, 1], int) == [99, 99]
    assert mod.get_integer([0, 1]) == [99, 99]
    assert mod.get_integer([0, 1]) == [99, 99]
    with pytest.raises(AssertionError) as err:
        _ = mod.get_real([0, 1])
    assert str(err.value).startswith("Invalid type in 'get_")
    assert mod.get_real([2]) == [0.99]
    #    print(mod.get_real([6]))
    assert mod.get_real([5])[0] == 1.0
    assert mod.get_real([5, 6]) == [1.0, math.radians(2)], f"Found {mod.get_real([5,6])}"
    assert mod.get_real([5, 6, 7]) == [1.0, math.radians(2), 3.0], f"Found {mod.get_real([5,6,7])}"


#         with pytest.raises( KeyError) as err:
#             vals = mod.get_real([5,8])
#         assert  str( err.exception).startswith("Variable with valueReference=8 does not exist in model My"))


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
    mod.set_integer([0, 1], [60, 61])
    assert mod.vars[0].getter() == 60
    assert mod.myInt == 60
    assert mod.vars[1].getter() == 61
    with pytest.raises(AssertionError) as err:
        mod.set_integer([6, 7], [2.0, "30 deg"])
    assert str(err.value).startswith("Invalid type in 'get_")
    #    print(
    mod.set_real([6, 7], [2.0, 0.5])  # "30 deg"])


def test_xml():
    Model.instances = []  # reset
    mod = Model("MyModel")
    myInt = Variable(
        mod,
        "myInt",
        description="A integer variable",
        causality="parameter",
        variability="fixed",
        value0="99%",
        rng=(0, "100%"),
        annotations=None,
        value_check=VarCheck.all,
    )
    myInt.to_xml()
    myNP2 = VariableNP(
        mod,
        "Test9",
        description="A NP variable with units included in initial values and partially fixed range",
        causality="input",
        variability="continuous",
        value0=("1m", "2deg", "3 deg"),
        rng=((0, "3m"), None, None),
    )
    myNP2.to_xml()


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
    myNP2.value = (3, 4, 5)


if __name__ == "__main__":
    test_var_check()
    test_spherical_cartesian()
    test_init()
    test_variable_np()
    test_dirty()
    test_var_ref()
    test_vars_iter()
    test_get()
    test_set()
    test_xml()
