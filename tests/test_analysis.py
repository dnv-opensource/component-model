# pyright: ignore[reportAttributeAccessIssue] # PythonFMU generates variable value objects using setattr()
import logging

import numpy as np
import pytest

from component_model.utils.analysis import extremum, extremum_series

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def test_extremum():
    t = [np.radians(10 * x) for x in range(100)]
    x = [np.cos(x) for x in t]
    e, p = extremum(t[0:3], x[0:3], 2e-3)  # allow a small error
    assert e == 1
    assert p[0] > -2e-3 and p[1] < 1 + 1e-6, (
        "Top of parabola somewhat to the left due to cos not exactly equal to 2.order"
    )
    # for i in range(100):
    #    print(i, t[i], x[i])
    e, p = extremum(t[17:20], x[17:20])
    assert e == -1 and abs(p[0] - np.pi) < 1e-10 and p[1] == -1
    ex = extremum_series(t, x, "all")
    assert len(ex) == 2
    assert np.allclose(ex[0], (12.566370614359142, 1.0))
    assert np.allclose(ex[1], (15.707963267948958, -1.0))


if __name__ == "__main__":
    retcode = pytest.main(["-rP -s -v", __file__])
    assert retcode == 0, f"Return code {retcode}"
    # test_extremum()
