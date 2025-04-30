# pyright: ignore[reportAttributeAccessIssue] # PythonFMU generates variable value objects using setattr()
import logging
import math
import xml.etree.ElementTree as ET  # noqa: N817
from enum import Enum

import numpy as np
import pytest
from pythonfmu.enums import Fmi2Causality as Causality  # type: ignore
from pythonfmu.enums import Fmi2Initial as Initial  # type: ignore
from pythonfmu.enums import Fmi2Variability as Variability  # type: ignore

from component_model.model import Model
from component_model.variable import (
    Check,
    Variable,
    VariableInitError,
    VariableRangeError,
    cartesian_to_spherical,
    spherical_to_cartesian,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class DummyModel(Model):
    def __init__(self, name, **kwargs):
        super().__init__(name=name, description="Just a dummy model to be able to do testing", **kwargs)
        # the following satisfies the linter. Variables are automatically registered through register_variable()
        self.int1: int
        self.int2: int
        self.float1: float
        self.enum1: Enum
        self.str1: str
        self.np1: np.ndarray
        self.np2: np.ndarray
        self.bool1: bool

    def do_step(self, current_time: float, step_size: float):
        return True


def arrays_equal(arr1: np.ndarray | tuple | list, arr2: np.ndarray | tuple | list, dtype="float", eps=1e-7):
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

    int1 = Variable(
        mod,
        "int1",
        description="A integer variable",
        causality="parameter",
        variability="fixed",
        start=99,
        rng=(0, 100),
        typ=int,
        annotations=None,
        value_check=Check.all,
    )
    assert hasattr(mod, "int1")
    int2 = Variable(
        mod,
        "int2",
        description="A integer variable without range checking",
        causality="input",
        variability="continuous",
        start=99,
        rng=(0, 100),
        typ=int,
        value_check=Check.none,
    )
    assert hasattr(mod, "int1")
    float1 = mod.add_variable(
        "float1",
        description="A float variable",
        causality="input",
        variability="continuous",
        start="99.0%",
        rng=(0.0, None),
    )
    assert hasattr(mod, "float1")
    enum1 = Variable(
        mod,
        "enum1",
        description="An enumeration variable",
        causality="output",
        variability="discrete",
        start=Causality.parameter,
    )
    assert hasattr(mod, "enum1")
    bool1 = Variable(
        mod,
        "bool1",
        description="A boolean variable",
        causality="parameter",
        variability="fixed",
        start=True,
    )
    assert hasattr(mod, "bool1")
    str1 = Variable(
        mod,
        "str1",
        description="A string variable",
        typ=str,
        causality="parameter",
        variability="fixed",
        start="Hello World!",
    )
    assert hasattr(mod, "str1")
    np1 = Variable(
        mod,
        "np1",
        description="A NP variable",
        causality="parameter",
        variability="fixed",
        start=("1.0m", "2deg", "3rad"),
        rng=((0, "3m"), ("1 deg", "5 deg"), (float("-inf"), "5rad")),
    )
    assert hasattr(mod, "np1")
    np2 = Variable(
        mod,
        "np2",
        description="A NP variable with on_set and on_step",
        causality="input",
        variability="continuous",
        start=("1.0", "2.0", "3.0"),
        rng=((0, float("inf")), (0, float("inf")), (0, float("inf"))),
        on_set=lambda val: 0.9 * val,
        on_step=lambda t, dt: mod.np2[0](dt * mod.np2[0]),
    )
    assert hasattr(mod, "np2")
    return (mod, int1, int2, float1, enum1, str1, np1, np2, bool1)


def test_init():
    mod = DummyModel("MyModel")
    (
        mod,
        int1,
        int2,
        float1,
        enum1,
        str1,
        np1,
        np2,
        bool1,
    ) = init_model_variables()

    assert int1.typ is int
    assert int1.description == "A integer variable"
    assert int1.causality == Causality.parameter
    assert int1.variability == Variability.fixed
    assert int1.initial == Initial.exact
    assert int1.check == Check.all
    # internally packed into tuple:
    assert int1.start == (99,)
    assert int1.range == ((0, 100),)
    assert int1.unit == ("dimensionless",)
    assert int1.display == (None,)
    assert int1.check_range([50])
    assert not int1.check_range([110])
    assert mod.int1 == 99, "Value directly accessible as model variable"
    mod.int1 = 110
    assert mod.int1 == 110, "Internal changes not range-checked!"
    with pytest.raises(VariableRangeError) as err:  # ... but getter() detects the range error
        _ = int1.getter()
    assert str(err.value) == "getter(): Value [110] outside range."
    assert mod.int1 == 110, f"Value {mod.int1} should still be unchanged"
    int1.setter([50])
    assert mod.int1 == 50, f"Value {mod.int1} changed back."
    mod.set_integer([mod.variable_by_name("int1").value_reference], [99])  # simulate setting from outside
    assert mod.get_integer([mod.variable_by_name("int1").value_reference]) == [99]

    assert float1.typ is float
    assert float1.causality == Causality.input
    assert float1.variability == Variability.continuous
    assert float1.initial is None, f"initial: {float1.initial}"
    assert float1.check == Check.all
    # internally packed into tuple:
    assert float1.start == (0.99,)
    assert float1.range == ((0, 99.0),), f"Range: {float1.range} in display units."
    assert float1.unit == ("dimensionless",)
    assert float1.display[0][0] == "percent", f"Display: {float1.display[0][0]}"
    assert float1.display[0][1](99) == 0.99, "Transform from dimensionless to percent"
    assert float1.display[0][2](0.99) == 99, "... and back."
    assert float1.check_range([0.5])
    assert not float1.check_range([1.0], disp=False), "Check as internal units"
    assert not float1.check_range([100.0]), "Check as display units"
    assert mod.float1 == 0.99, "Value directly accessible as model variable"
    mod.float1 = 1.0
    assert mod.float1 == 1.0, "Internal changes not range-checked!"
    with pytest.raises(VariableRangeError) as err:  # ... but getter() detects the range error
        _ = float1.getter()
    assert str(err.value) == "getter(): Value [100.0] outside range."
    assert mod.float1 == 1.0, f"Value {mod.float1} should still be unchanged"
    float1.setter([50])
    assert mod.float1 == 0.5, f"Value {mod.float1} changed back."
    mod.set_real([mod.variable_by_name("float1").value_reference], [99])  # simulate setting from outside
    assert mod.get_real([mod.variable_by_name("float1").value_reference]) == [99]

    assert isinstance(enum1.start[0], Enum), f"Type: {enum1.typ}"
    assert enum1.typ is not None and issubclass(enum1.typ, Enum), "Enums are always derived"
    assert enum1.causality == Causality.output
    assert enum1.variability == Variability.discrete
    assert enum1.initial == Initial.calculated, f"initial: {enum1.initial}"
    assert enum1.check == Check.all
    # internally packed into tuple:
    assert enum1.start == (Causality.parameter,)
    assert enum1.range == ((0, 4),), f"Range: {enum1.range}"
    assert enum1.unit == ("dimensionless",)
    assert enum1.display[0] is None, f"Display: {enum1.display[0]}"
    assert enum1.check_range([1])
    assert not enum1.check_range([7])
    assert mod.enum1 == Causality.parameter, f"Value {mod.enum1} directly accessible as model variable"
    mod.enum1 = Causality.input
    assert mod.enum1 == Causality.input, "Not possible to supply a wrong value with the right type!"
    enum1.setter([Causality.local])
    assert enum1.getter() == [4], f"Value {enum1.getter()}. Translated to int!"
    mod.set_integer([mod.variable_by_name("enum1").value_reference], [2])  # simulate setting from outside
    assert mod.get_integer([mod.variable_by_name("enum1").value_reference]) == [2]

    assert bool1.typ is bool
    assert bool1.causality == Causality.parameter
    assert bool1.variability == Variability.fixed
    assert bool1.initial == Initial.exact
    assert bool1.check == Check.all
    # internally packed into tuple:
    assert bool1.start == (True,)
    assert bool1.range == ((False, True),)
    assert bool1.unit == ("dimensionless",)
    assert bool1.display == (None,)
    assert bool1.check_range([True])
    assert bool1.check_range([100.5]), "Any number will work"
    assert not bool1.check_range("Hei"), "But non-numbers are rejected"
    assert mod.bool1, "Value directly accessible as model variable"
    mod.bool1 = 100  # type: ignore # just to demonstrate that range checking is not done on internal variables
    assert mod.bool1 == 100, "Internal changes not range-checked!"
    assert bool1.getter(), "Converted in getter()"
    bool1.setter([False])
    assert not mod.bool1, f"Value {mod.bool1} changed."
    mod.set_boolean([mod.variable_by_name("bool1").value_reference], [True])  # simulate setting from outside
    assert mod.get_boolean([mod.variable_by_name("bool1").value_reference]) == [True]

    assert str1.typ is str
    assert str1.causality == Causality.parameter
    assert str1.variability == Variability.fixed
    assert str1.initial == Initial.exact, f"initial: {str1.initial}"
    assert str1.check == Check.all
    # internally packed into tuple:
    assert str1.start == ("Hello World!",)
    assert str1.range == (("", ""),), f"Range: {str1.range}. Basically irrelevant"
    assert str1.unit == ("dimensionless",), f"Unit {str1.unit}"
    assert str1.display[0] is None, f"Display: {str1.display[0]}"
    assert str1.check_range([0.5]), "Everything is ok"
    assert mod.str1 == "Hello World!", f"Value {mod.str1} directly accessible as model variable"
    mod.str1 = 1.0  # type: ignore # intentional misuse
    assert mod.str1 == 1.0, f"Not converted to str when internally set: {mod.str1}"
    assert isinstance(str1.getter()[0], str), f"getter() should convert to str. Got {type(str1.getter()[0])}"
    str1.setter(["Hei"])
    assert mod.str1 == "Hei", f"New value {mod.str1}."
    mod.set_string([mod.variable_by_name("str1").value_reference], ["Hello"])  # simulate setting from outside
    assert mod.get_string([mod.variable_by_name("str1").value_reference]) == ["Hello"]

    assert np1.typ is float
    assert np1 == mod.variable_by_name("np1")
    assert np1.description == "A NP variable"
    assert mod.variable_by_name("np1[1]") == mod.variable_by_name("np1"), "Returns always the parent"
    assert np1.causality == Causality.parameter
    assert np1.variability == Variability.fixed
    assert np1.initial == Initial.exact
    assert np1.check == Check.all
    # internally packed into tuple:
    assert np1.start == (1, math.radians(2), 3)
    tuples_nearly_equal(np1.range, ((0, 3), (1, 5), (float("-inf"), 5)))
    assert not np1.check_range([5.1], idx=1), "Checks performed on display units!"
    assert not np1.check_range([0.9], idx=1), "Checks performed on display units!"
    assert np1.unit == ("meter", "radian", "radian"), f"Units: {np1.unit}"
    assert isinstance(np1.display, tuple) and len(np1.display) == 3, "Tuple of length 3 expected"
    assert np1.display[0] is None
    assert np1.display[1][0] == "degree"
    assert np1.display[2] is None
    assert np1.check_range((2, 3.5, 4.5))
    assert not np1.check_range((2, 3.5, 6.3), -1), f"Range is {np1.range}"
    assert mod.np1[1] == math.radians(2), "Value directly accessible as model variable"
    mod.np1[1] = -1.0
    assert mod.np1[1] == -1.0, "Internal changes not range-checked!"
    with pytest.raises(VariableRangeError) as err:  # ... but getter() detects the range error
        _ = np1.getter()
    assert str(err.value) == "getter(): Value [1.0, -57.29577951308233, 3.0] outside range."
    assert mod.np1[1] == -1.0, f"Value {mod.np1} should still be unchanged"
    mod.np1 = np.array((1.5, 2.5, 3.5), float)
    assert np.linalg.norm(mod.np1) == math.sqrt(1.5**2 + 2.5**2 + 3.5**2), "np calculations are done on value"
    np1.setter((1.0, 1.0, 1.0))
    arrays_equal(mod.np1, (1.0, math.radians(1.0), 1.0))
    res = np1.getter()
    assert isinstance(res, (list, np.ndarray))
    arrays_equal(res, [1.0, 1.0, 1.0])  # getter shows display units
    vr0 = mod.variable_by_name("np1").value_reference
    mod.set_real([vr0, vr0 + 1, vr0 + 2], [2.0, 2.0, 2.0])  # simulate setting from outside
    arrays_equal(mod.get_real((vr0, vr0 + 1, vr0 + 2)), [2.0, 2.0, 2.0])
    arrays_equal(mod.get_real([vr0, vr0 + 1, vr0 + 2]), [2.0, 2.0, 2.0])  # array not changed by getter (need copy)

    with pytest.raises(KeyError) as err2:
        _ = Variable(
            mod,
            "int1",
            description="An integer variable with a non-unique name",
            causality="input",
            variability="continuous",
            typ=int,
            start="99.9%",
            rng=(0, "100%"),
            annotations=None,
        )
    assert err2.value.args[0] == "Variable int1 already used as index 0 in model MyModel"

    with pytest.raises(VariableInitError) as err3:
        int1 = Variable(
            mod,
            "bool1",
            description="A second integer variable with erroneous range",
            causality="parameter",
            variability="fixed",
            start="99",
            rng=(),
            annotations=None,
            typ=int,
        )
    assert err3.value.args[0].startswith("Range must be specified for int variable")
    assert float1.range[0][1] == 99.0
    assert enum1.range[0] == (0, 4)
    assert enum1.check_range([Causality.parameter])
    assert str1.range == (("", ""),), "Just a placeholder. Range of str is not checked"
    assert bool1.typ is bool


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
    with pytest.raises(VariableRangeError) as err2:  # causes an error
        _ = int1.getter()
    assert err2.value.args[0] == "getter(): Value [6.0] outside range."
    float1 = Variable(mod, "float1", start=1, typ=float)  # explicit type
    assert float1.range == ((float("-inf"), float("inf")),), "Auto_extreme. Same as rng=()"
    float2 = Variable(mod, "float2", start=1.0, rng=None)  # implicit type through start value and no range
    assert float2.range == ((1.0, 1.0),), "No range."
    with pytest.raises(VariableRangeError) as err3:
        float2.setter([2.0])
    assert err3.value.args[0] == "set(): values [2.0] outside range."

    np1 = Variable(mod, "np1", start=("1.0m", 2, 3), rng=((0, "3m"), None, tuple()))
    assert np1.range == ((0.0, 3.0), (2.0, 2.0), (float("-inf"), float("inf")))


def test_dirty():
    """Test the dirty mechanism"""
    mod = DummyModel("MyModel2", instance_name="MyModel2")
    np1 = Variable(
        mod,
        "np1",
        description="A NP variable",
        causality="parameter",
        variability="fixed",
        start=("1.0m", "2deg", "3rad"),
        on_set=lambda x: 0.5 * x,
        rng=((0, "3m"), (0, float("inf")), (float("-inf"), "5rad")),
    )
    assert np1.typ is float, f"Type {np1.typ}"
    np1.setter(np.array((2, 1, 4), float))
    assert np1 not in mod.dirty, "Not dirty, because the whole variable was changed"
    arrays_equal(mod.np1, [0.5 * 2.0, 0.5 * math.radians(1), 0.5 * 4])  # ... and on_set has been run
    mod.set_real([1], [9.9])
    assert np1 in mod.dirty, "Dirty. on_set has not been run."
    res = np1.getter()
    assert isinstance(res, list), f"List expected. Got {np1.getter()}"
    arrays_equal(res, [1, 9.9, 2])  # on_set not yet run
    mod.dirty_do()
    res = np1.getter()
    assert isinstance(res, list), f"List expected. Got {np1.getter()}"
    arrays_equal(res, [0.5 * 1, 0.5 * 9.9, 0.5 * 2])  # on_set run


def test_var_ref():
    Model.instances = []  # reset
    (
        mod,
        int1,
        int2,
        float1,
        enum1,
        str1,
        np1,
        np2,
        bool1,
    ) = init_model_variables()
    assert mod.vars[1].name == "int2"
    assert mod.vars[6].name == "np1"
    assert mod.vars[7] is None, "a sub-element"
    var, sub = mod.ref_to_var(7)
    assert var.name == "np1"
    assert sub == 1
    assert mod.variable_by_name("int2").name == "int2"
    assert mod.variable_by_name("int2").value_reference == 1
    # mod.variable_by_value(mod.int1) deleted


def test_vars_iter():
    Model.instances = []  # reset
    (
        mod,
        int1,
        int2,
        float1,
        enum1,
        str1,
        np1,
        np2,
        bool1,
    ) = init_model_variables()
    assert list(mod.vars_iter(float)) == [float1, np1, np2]
    assert list(mod.vars_iter(float))[0].name == "float1"
    assert list(mod.vars_iter(float))[1].name == "np1"
    assert list(mod.vars_iter(key=Variability.discrete))[0].name == "enum1"
    assert list(mod.vars_iter(key=Causality.input))[1].name == "float1"
    assert (
        list(mod.vars_iter(key=lambda x: x.causality == Causality.input or x.causality == Causality.output))[2].name
        == "enum1"
    )


def test_get():
    Model.instances = []  # reset
    (
        mod,
        int1,
        int2,
        float1,
        enum1,
        str1,
        np1,
        np2,
        bool1,
    ) = init_model_variables()
    # print( "".join( str(i)+":"+mod.vars[i].name+", " for i in range( len(mod.vars)) if mod.vars[i] is not None))
    assert mod._get([0, 1], int) == [99, 99]
    assert mod.get_integer([0, 1]) == [99, 99]
    with pytest.raises(AssertionError) as err:
        _ = mod.get_real([0, 1])
    assert err.value.args[0].startswith("Invalid type in 'get_")
    assert mod.get_real([2]) == [99.0], f"Got value {mod.get_real([2])} (converted to %)"
    assert mod.get_string([5])[0] == "Hello World!"
    assert abs(mod.get_real([7])[0] - 2.0) < 1e-14, "Second element of compound variable"
    assert mod.get_real([6])[0] == 1.0
    assert mod.vars[6].name == "np1"
    var, sub = mod.ref_to_var(7)
    assert var.name == "np1" and sub == 1, "Second element of NP variable"
    assert len(var) == 3
    assert mod.variable_by_name("np1").value_reference == 6
    arrays_equal(mod.get_real([6, 7, 8]), [1.0, 2.0, 3.0])  # translated back to degrees
    with pytest.raises(AssertionError) as err:
        _ = mod.get_real([9, 12])
    assert err.value.args[0] == "valueReference=12 does not exist in model MyModel"


def test_set():
    Model.instances = []  # reset
    (
        mod,
        int1,
        int2,
        float1,
        enum1,
        str1,
        np1,
        np2,
        bool1,
    ) = init_model_variables()
    # print( "".join( str(i)+":"+mod.vars[i].name+", " for i in mod.vars if mod.vars[i] is not None))
    mod.set_integer([0, 1], [60, 61])
    assert mod.vars[0].getter() == [60], f"Found {mod.vars[0].getter()}"
    assert mod.int1 == 60
    assert mod.vars[1].getter() == [61], f"Found {mod.vars[99].getter()}"
    with pytest.raises(AssertionError) as err:
        mod.set_integer([6, 7], [2.0, "30 deg"])  # type: ignore # we want to produce an error!
    assert str(err.value) == "Invalid type in 'set_<class 'int'>'. Found variable np1 with type <class 'float'>"
    mod.set_real([6, 7], [2.0, 3.0])  # "3 deg"])


# @pytest.mark.skip()
def test_xml():
    Model.instances = []  # reset
    mod = DummyModel("MyModel")
    np2 = Variable(
        mod,
        "Test9",
        description="A NP variable ...",
        causality="input",
        variability="continuous",
        start=("1m", "2deg", "3 deg"),
        rng=((0, "3m"), None, None),
    )
    lst = np2.xml_scalarvariables()
    assert len(lst) == 3
    expected = '<ScalarVariable name="Test9[0]" valueReference="0" description="A NP variable ..." causality="input" variability="continuous"><Real start="1.0" min="0.0" max="3.0" unit="meter" /></ScalarVariable>'
    assert ET.tostring(lst[0], encoding="unicode") == expected, ET.tostring(lst[0], encoding="unicode")
    expected = '<ScalarVariable name="Test9[1]" valueReference="1" description="A NP variable ..." causality="input" variability="continuous"><Real start="0.03490658503988659" min="1.9999999999999993" max="2.0000000000000013" unit="radian" displayUnit="degree" /></ScalarVariable>'
    assert ET.tostring(lst[1], encoding="unicode") == expected, ET.tostring(lst[1], encoding="unicode")
    expected = '<ScalarVariable name="Test9[2]" valueReference="2" description="A NP variable ..." causality="input" variability="continuous"><Real start="0.05235987755982989" min="2.9999999999999996" max="3.0000000000000013" unit="radian" displayUnit="degree" /></ScalarVariable>'
    assert ET.tostring(lst[2], encoding="unicode") == expected, ET.tostring(lst[2], encoding="unicode")

    int1 = Variable(
        mod,
        "int1",
        description="A integer variable",
        causality="parameter",
        variability="fixed",
        start="99%",
        rng=(0, "100%"),
        annotations=None,
        value_check=Check.all,
    )
    lst = int1.xml_scalarvariables()
    expected = '<ScalarVariable name="int1" valueReference="3" description="A integer variable" causality="parameter" variability="fixed" initial="exact"><Real start="0.99" min="0.0" max="100.0" unit="dimensionless" displayUnit="percent" /></ScalarVariable>'
    found = ET.tostring(lst[0], encoding="unicode")
    assert found == expected, f"\nFound   :{found}\nExpected:{expected}"


def test_on_set():
    (
        mod,
        int1,
        int2,
        float1,
        enum1,
        str1,
        np1,
        np2,
        bool1,
    ) = init_model_variables()
    # print( "".join( str(i)+":"+mod.vars[i].name+", " for i in range( len(mod.vars)) if mod.vars[i] is not None))
    arrays_equal(mod.np2, (1, 2, 3))
    np2.setter(np.array((4, 5, 6), float), idx=-1)
    arrays_equal(mod.np2, (0.9 * 4, 0.9 * 5, 0.9 * 6))  # on_set run, because whole array is set
    mod.set_real([10, 11], [7, 8])
    arrays_equal(mod.np2, (0.9 * 4, 7, 8))
    mod.dirty_do()
    arrays_equal(mod.np2, (0.9 * 0.9 * 4, 0.9 * 7, 0.9 * 8))


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
