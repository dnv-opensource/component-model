import logging
from math import isinf

import numpy as np
from scipy.spatial.transform import Rotation as Rot

logger = logging.getLogger(__name__)


# Utility functions for handling special variable types
def spherical_to_cartesian(vec: np.ndarray | tuple[float, ...], deg: bool = False) -> np.ndarray:
    """Turn spherical vector 'vec' (defined according to ISO 80000-2 (r,polar,azimuth)) into cartesian coordinates."""
    if deg:
        theta = np.radians(vec[1])
        phi = np.radians(vec[2])
    else:
        theta = vec[1]
        phi = vec[2]
    sinTheta = np.sin(theta)
    cosTheta = np.cos(theta)
    sinPhi = np.sin(phi)
    cosPhi = np.cos(phi)
    r = vec[0]
    return np.array((r * sinTheta * cosPhi, r * sinTheta * sinPhi, r * cosTheta))


def cartesian_to_spherical(vec: np.ndarray | tuple[float, ...], deg: bool = False) -> np.ndarray:
    """Turn the vector 'vec' given in cartesian coordinates into spherical coordinates.
    (defined according to ISO 80000-2, (r, polar, azimuth)).
    """
    r = np.linalg.norm(vec)
    if vec[0] == vec[1] == 0:
        if vec[2] == 0:
            return np.array((0, 0, 0), float)
        else:
            return np.array((r, 0, 0), float)
    elif deg:
        return np.array((r, np.degrees(np.arccos(vec[2] / r)), np.degrees(np.arctan2(vec[1], vec[0]))), float)
    else:
        return np.array((r, np.arccos(vec[2] / r), np.arctan2(vec[1], vec[0])), float)


def cartesian_to_cylindrical(vec: np.ndarray | tuple[float, ...], deg: bool = False) -> np.ndarray:
    """Turn the vector 'vec' given in cartesian coordinates into cylindrical coordinates.
    (defined according to ISO, (r, phi, z), with phi right-handed wrt. x-axis).
    """
    phi = np.arctan2(vec[1], vec[0])
    if deg:
        phi = np.degrees(phi)
    return np.array((np.sqrt(vec[0] * vec[0] + vec[1] * vec[1]), phi, vec[2]), float)


def cylindrical_to_cartesian(vec: np.ndarray | tuple[float, ...], deg: bool = False) -> np.ndarray:
    """Turn cylinder coordinate vector 'vec' (defined according to ISO (r,phi,z)) into cartesian coordinates.
    The angle phi is measured with respect to x-axis, right hand.
    """
    phi = np.radians(vec[1]) if deg else vec[1]
    return np.array((vec[0] * np.cos(phi), vec[0] * np.sin(phi), vec[2]), float)


def euler_rot_spherical(
    rpy: tuple[float, ...] | list[float] | Rot,
    vec: tuple[float, ...] | list[float] | None = None,
    seq: str = "XYZ",  # sequence of axis of rotation as defined in scipy Rotation object
    degrees: bool = False,
) -> np.ndarray:
    """Rotate the spherical vector vec using the Euler angles (yaw,pitch,roll).

    Args:
        rpy (Sequence|Rotation): The sequence of (yaw,pitch,roll) Euler angles or the pre-calculated Rotation object
        vec: (Sequence): the spherical vector to be rotated
           None: Use unit vector in z-direction, i.e. (1,0,0)
           2-sequence: only polar and azimuth provided and returned => (polar,azimuth)
           3-sequence: (r,polar,azimuth)
        seq (str) = 'XYZ': Sequence of rotations as defined in scipy.spatial.transform.Rotation.from_euler()
        degrees (bool): angles optionally provided in degrees. Default: radians
    Returns:
        The rotated vector in spherical coordinates (radius only if 3-vector is provided)
    """
    if isinstance(rpy, Rot):
        r = rpy
    elif isinstance(rpy, (tuple, list, np.ndarray)):
        r = Rot.from_euler(seq, (rpy[0], rpy[1], rpy[2]), degrees)  # 0: roll, 1: pitch, 2: yaw
    else:
        logger.critical(f"Unknown object {rpy} to rotate")
        raise NotImplementedError(f"Unknown object {rpy} to rotate") from None
    radius = 0.0  # avoid spurious unbound error
    if vec is None:
        tp = [0.0, 0.0]
    else:  # explicit vector provided
        if len(vec) == 3:
            radius = vec[0]
            tp = list(vec[1:])
        else:
            tp = list(vec)
        if degrees:
            tp = [np.radians(x) for x in tp]
    st = np.sin(tp[0])
    # rotate the cartesian vector (r is definitely not a list, even if pyright might think so)
    x = r.apply((st * np.cos(tp[1]), st * np.sin(tp[1]), np.cos(tp[0])))  # type: ignore[reportAttributeAccessIssue]
    x2 = x[2]
    if abs(x2) < 1.0:
        pass
    elif abs(x2 - 1.0) < 1e-10:
        x2 = 1.0
    elif abs(x2 + 1.0) < 1e-10:
        x2 = -1.0
    else:
        logger.critical(f"Invalid argument {x2} for arccos calculation")
        raise ValueError(f"Invalid argument {x2} for arccos calculation") from None
    if abs(x[0]) < 1e-10 and abs(x[1]) < 1e-10:  # define the normally undefined arctan
        phi = 0.0
    else:
        phi = np.arctan2(x[1], x[0])
    if vec is not None and len(vec) == 3:
        return np.array((radius, np.arccos(x2), phi), float)  # return the spherical vector
    else:
        return np.array((np.arccos(x2), phi), float)  # return only the direction


def rot_from_spherical(vec: tuple[float, ...] | list[float] | np.ndarray, degrees: bool = False) -> Rot:
    """Return a scipy Rotation object from the spherical coordinates vec,
    i.e. the rotation which turns a vector along the z-axis into vec.

    Args:
        vec (Sequence | np.ndarray): a spherical vector as 3D or 2D (radius omitted)
        degrees (bool): optional possibility to provide angles in degrees
    """
    angle = vec[1:] if len(vec) == 3 else vec
    return Rot.from_rotvec((0.0, 0.0, angle[1]), degrees) * Rot.from_rotvec((0.0, angle[0], 0.0), degrees)


def rot_from_vectors(vec1: np.ndarray, vec2: np.ndarray) -> Rot:
    """Find the rotation object which rotates vec1 into vec2. Lengths of vec1 and vec2 shall be equal."""
    n = np.linalg.norm(vec1)
    assert abs(n - np.linalg.norm(vec2)) < 1e-10, f"Vectors len({vec1}={n} != len{vec2}. Cannot rotate into each other"
    if abs(n - 1.0) > 1e-10:
        vec1 /= n
        vec2 /= n
    _c = vec1.dot(vec2)  # type: ignore
    if abs(_c + 1.0) < 1e-10:  # vectors are exactly opposite to each other
        imax, vmax, _sum = (-1, float("-inf"), 0.0)
        for k, v in enumerate(vec1):
            if isinf(vmax) or abs(v) > abs(vmax):
                imax, vmax = (k, v)
            _sum += v
        vec = np.zeros(3)
        vec[imax] = -(_sum - vmax) / vmax
        vec[imax + 1 if imax < 2 else 0] = 0.0
        i_remain = imax + 2 if imax < 1 else 1
        vec[i_remain] = np.sqrt(1.0 / (1 + (vec1[i_remain] / vmax) ** 2))
        return Rot.from_rotvec(np.pi * vec)
    else:
        x = np.cross(vec1, vec2)
        vx = np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]], float)
        # print(vec1, vec2, _c, x,'\n',vx,'\n',np.matmul(vx,vx))
        return Rot.from_matrix(np.identity(3) + vx + np.matmul(vx, vx) / (1 + _c))


def spherical_unique(vec: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    if len(vec) == 3:
        if abs(vec[0]) < eps:
            return np.array((0, 0, 0), float)
        elif vec[0] < 0:
            return np.append(-vec[0], spherical_unique(np.array((vec[1] + np.pi, vec[2]), float)))
        else:
            return np.append(vec[0], spherical_unique(vec[1:], eps))
    else:
        if abs(vec[0]) < eps:
            return np.array((0, 0), float)
        elif 0 <= vec[0] <= np.pi and 0 <= vec[1] <= 2 * np.pi:
            return vec
        # angles not in unique range
        theta = vec[0]
        phi = vec[1]
        _2pi = np.pi + np.pi
        while theta > np.pi:
            theta -= _2pi
        while theta < -np.pi:
            theta += _2pi
        if theta < 0:
            theta = -theta
            phi += np.pi
        while phi < 0:
            phi += _2pi
        while phi >= _2pi:
            phi -= _2pi
        return np.array((theta, phi), float)


def quantity_direction(quantity_direction: tuple[float, ...], spherical: bool = False, deg: bool = False) -> np.ndarray:
    """Turn a 4-tuple, consisting of quantity (float) and a direction 3-vector to a direction 3-vector,
    where the norm denotes the direction and the length denotes the quantity.
    The return vector is always a cartesian vector.

    Args:
        quantity_direction (tuple): a 4-tuple consisting of the desired length of the resulting vector (in standard units (m or m/s))
           and the direction 3-vector (in standard units)
        spherical (bool)=False: Optional possibility to provide the input direction vector in spherical coordinates
        deg (bool)=False: Optional possibility to provide the input angle (of spherical coordinates) in degrees. Only relevant if spherical=True
    """
    if quantity_direction[0] < 1e-15:
        return np.array((0, 0, 0), float)
    if spherical:
        direction = spherical_to_cartesian(quantity_direction[1:], deg)  # turn to cartesian coordinates, if required
    else:
        direction = np.array(quantity_direction[1:], float)
    n = np.linalg.norm(direction)  # normalize
    return quantity_direction[0] / n * direction


def normalized(vec: np.ndarray) -> np.ndarray:
    """Return the normalized vector. Helper function."""
    assert len(vec) == 3, f"{vec} should be a 3-dim vector"
    norm = np.linalg.norm(vec)
    assert norm > 0, f"Zero norm detected for vector {vec}"
    return vec / norm
