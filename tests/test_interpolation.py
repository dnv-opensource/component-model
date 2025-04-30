import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.interpolate import CubicSpline, make_interp_spline


def test_np_linear(show: bool = False):
    x = np.linspace(0, 10, num=11)
    y = np.cos(-(x**2) / 9.0)

    xnew = np.linspace(0, 10, num=1001)
    ynew = np.interp(xnew, x, y)
    if show:
        plt.plot(xnew, ynew, "-", label="linear interp")
        plt.plot(x, y, "o", label="data")
        plt.legend(loc="best")
        plt.show()


def test_cubic_spline(show: bool = False):
    spl = CubicSpline([1, 2, 3, 4, 5, 6], [1, 4, 8, 16, 25, 36])
    print("SPLINE", spl(2.5))
    print("EXTRAPOLATE", spl(7.0), 7**2)


def bspline(k: int = 1, show: bool = False):
    x = np.linspace(0, 10, num=11)
    y = np.cos(-(x**2) / 9.0)
    bspl = make_interp_spline(x, y, k=k)
    xnew = np.linspace(0, 10, num=1001)
    ynew = bspl(xnew)
    if show:
        plt.plot(xnew, ynew, "-", label="bsp")
        plt.plot(x, y, "o", label="data")
        plt.legend(loc="best")
        plt.show()


def test_bspline_0(show: bool = False):
    bspline(0, show)


def test_bspline_1(show: bool = False):
    bspline(1, show)


def test_bspline_2(show: bool = False):
    bspline(2, show)


def test_bspline_3(show: bool = False):
    bspline(3, show)


def bspline_multi(k: int = 1, show: bool = False):
    x = np.linspace(0, 10, num=11)
    y = np.array([[np.cos(-(u**2) / 9.0), np.sin(-(u**2) / 9.0)] for u in x])
    print("Y", y)
    bspl = make_interp_spline(x, y, k=k)
    xnew = np.linspace(0, 10, num=1001)
    ynew = bspl(xnew)
    if show:
        plt.plot(xnew, ynew, "-", label="bsp")
        plt.plot(x, y, "o", label="data")
        plt.legend(loc="best")
        plt.show()


def test_bspline_multi_3(show: bool = False):
    bspline_multi(3, show)


if __name__ == "__main__":
    retcode = pytest.main(["-rA", "-v", __file__, "--show", "True"])
    assert retcode == 0, f"Non-zero return code {retcode}"

    # test_bspline_3( show=True)
    # test_bspline_multi_3( show=True)
