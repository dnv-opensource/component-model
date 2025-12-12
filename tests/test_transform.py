import logging

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rot

from component_model.utils.transform import (
    cartesian_to_spherical,
    euler_rot_spherical,
    normalized,
    rot_from_spherical,
    rot_from_vectors,
    spherical_to_cartesian,
    spherical_unique,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


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


def test_spherical_unique():
    def do_test(x0: tuple, x1: tuple):
        if len(x0) == 3:
            _unique = spherical_unique(np.append(x0[0], np.radians(x0[1:])))
            unique = np.append(_unique[0], np.degrees(_unique[1:]))
        else:
            unique = np.degrees(spherical_unique(np.radians(x0)))
        assert np.allclose(unique, x1), f"{x0} -> {list(unique)} != {list(x1)}"

    do_test((0, 99), (0.0, 0.0))
    do_test((-1, 10, 20), (1, 180 - 10.0, 180 + 20))
    do_test((10, 300), (10, 300))
    do_test((190, 300), (170.0, 300.0 + 180 - 360))
    do_test((170, 300), (170.0, 300.0))
    do_test((170, 720), (170.0, 0.0))


def test_rot_from_spherical():
    assert np.allclose(rot_from_spherical((0, 0)).as_matrix(), Rot.identity().as_matrix())
    assert np.allclose(rot_from_spherical((90, 0), True).apply((0, 0, 1)), (1.0, 0, 0))
    assert np.allclose(rot_from_spherical((0, 90), True).apply((0, 0, 1)), (0.0, 0, 1.0))
    assert np.allclose(rot_from_spherical((0, 90), True).apply((1.0, 0, 0)), (0.0, 1.0, 0.0))
    assert np.allclose(rot_from_spherical((1.0, 45, 45), True).apply((0.0, 0, 1.0)), (0.5, 0.5, np.sqrt(2) / 2))
    down = rot_from_spherical((180, 0), True)
    print("down+2", down, down * rot_from_spherical((2, 0), True))


def test_rot_from_vectors():
    def do_check(vec1: tuple | list | np.ndarray, vec2: tuple | list | np.ndarray):
        v1 = np.array(vec1, float)
        v2 = np.array(vec2, float)
        r = rot_from_vectors(v1, v2)
        v = r.apply(v1)
        assert np.allclose(v2, v), f"{r.as_matrix} does not turn {v1} into {v2}"

    do_check((1, 0, 0), (-1, 0, 0))
    return
    do_check((1, 0, 0), (0, 1, 0))
    rng = np.random.default_rng(12345)
    for _i in range(100):
        v1 = normalized(rng.random(3))
        do_check(v1, -v1)  # opposite vectors
        do_check(v1, v1)  # identical vectors
    for _i in range(1000):
        do_check(normalized(rng.random(3)), normalized(rng.random(3)))


def test_euler_rot_spherical():
    """Test euler rotations.
    Note: We use XYZ + (roll, pitch, yaw) convention Tait-Brian."""
    _re = Rot.from_euler("zyx", (20, 40, 60), degrees=True)  # extrinsic rotation
    _ri = Rot.from_euler("XYZ", (60, 40, 20), degrees=True)  # intrinsic rotation
    assert np.allclose(_re.as_matrix(), _ri.as_matrix()), "Rotation matrices for extrinsic == intrisic+reversed"
    _re_inv = Rot.from_euler("xyz", (-60, -40, -20), degrees=True)
    assert np.allclose(_re.as_matrix(), _re_inv.as_matrix().transpose()), "_re_inv is inverse to _re"
    assert np.allclose(_re.as_matrix() @ _re_inv.as_matrix(), Rot.identity(3).as_matrix()), "_re_inv is the inverse."

    assert np.allclose(Rot.from_euler("XYZ", (90, 0, 0), degrees=True).apply((1, 0, 0)), (1, 0, 0)), "Roll invariant x"
    assert np.allclose(Rot.from_euler("XYZ", (90, 0, 0), degrees=True).apply((0, 1, 0)), (0, 0, 1)), (
        "Roll y(SB) -> z(down)"
    )
    assert np.allclose(Rot.from_euler("XYZ", (90, 0, 0), degrees=True).apply((0, 0, -1)), (0, 1, 0)), (
        "Roll -z(up) -> y(SB)"
    )
    assert np.allclose(Rot.from_euler("XYZ", (0, 90, 0), degrees=True).apply((1, 0, 0)), (0, 0, -1)), (
        "Pitch x(FW) -> -z(up)"
    )
    assert np.allclose(Rot.from_euler("XYZ", (0, 90, 0), degrees=True).apply((0, 1, 0)), (0, 1, 0)), "Pitch invariant y"
    assert np.allclose(Rot.from_euler("XYZ", (0, 90, 0), degrees=True).apply((0, 0, 1)), (1, 0, 0)), (
        "Pitch z(down) -> x(FW)"
    )
    assert np.allclose(Rot.from_euler("XYZ", (0, 0, 90), degrees=True).apply((1, 0, 0)), (0, 1, 0)), (
        "Yaw x(FW) -> y(SB)"
    )
    assert np.allclose(Rot.from_euler("XYZ", (0, 0, 90), degrees=True).apply((0, 1, 0)), (-1, 0, 0)), (
        "Yaw y(SB) -> -x(BW)"
    )
    assert np.allclose(Rot.from_euler("XYZ", (0, 0, 90), degrees=True).apply((0, 0, 1)), (0, 0, 1)), "Yaw invariant z"
    assert np.allclose(np.degrees(euler_rot_spherical((90, 0, 0), (90, 90), degrees=True)), (0, 0)), "Roll y -> z"
    assert np.allclose(np.degrees(euler_rot_spherical((0, 90, 0), (0, 0), degrees=True)), (90, 0)), "Pitch z -> x"
    assert np.allclose(np.degrees(euler_rot_spherical((0, 0, 90), (90, 0), degrees=True)), (90, 90)), "Yaw x -> y"


def test_euler_rot():
    """Test general issues about 3D rotations."""
    _rot = Rot.from_euler("XYZ", (90, 0, 0), degrees=True)  # roll 90 deg
    assert np.allclose(_rot.apply((0, 0, 1)), (0, -1, 0)), "z -> -y"
    _rot2 = Rot.from_euler("XYZ", (90, 0, 0), degrees=True) * _rot  # another 90 deg in same direction
    assert np.allclose(_rot2.apply((0, 0, 1)), (0, 0, -1)), "z -> -z"
    assert np.allclose(_rot2.apply((0, 0, 1)), Rot.from_euler("XYZ", (180, 0, 0), degrees=True).apply((0, 0, 1))), (
        "Angles added"
    )
    _rot2 = _rot.from_euler("XYZ", (0, 90, 0), degrees=True) * _rot  # + pitch 90 deg
    print(_rot2.as_euler(seq="XYZ", degrees=True))
    with pytest.warns(UserWarning, match="Gimbal lock detected"):
        print(Rot.from_euler("XYZ", (90, 90, 0), degrees=True).as_euler(seq="XYZ", degrees=True))
    _rot3 = Rot.from_euler("XYZ", (0, 0, 90), degrees=True) * _rot2  # +yaw 90 deg
    assert np.allclose(_rot3.apply((1, 0, 0)), (0, 0, -1))
    assert np.allclose(_rot3.apply((0, 1, 0)), (0, 1, 0))
    assert np.allclose(_rot3.apply((0, 0, 1)), (1, 0, 0))
    assert np.allclose(np.cross(_rot3.apply((1, 0, 0)), _rot3.apply((0, 1, 0))), _rot3.apply((0, 0, 1))), (
        "Still right-hand"
    )


def test_normalized():
    assert np.allclose(normalized(np.array((1, 0, 0), float)), (1, 0, 0))
    assert np.allclose(normalized(np.array((1, 1, 1), float)), np.array((1, 1, 1), float) / np.sqrt(3))


if __name__ == "__main__":
    retcode = 0  # pytest.main(["-rP -s -v", __file__])
    assert retcode == 0, f"Return code {retcode}"
    # test_spherical_cartesian()
    # test_spherical_unique()
    # test_rot_from_spherical()
    # test_rot_from_vectors()
    # test_euler_rot_spherical()
    test_euler_rot()
    # test_normalized()
