import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.integrate import simpson, solve_ivp


def do_show(
    time: list[float],
    y0: list[float],
    y1: list[float],
    compare1: list[float] | np.ndarray | None = None,
    compare2: list[float] | np.ndarray | None = None,
    z_label: str = "position",
    v_label: str = "speed",
):
    fig, ax = plt.subplots()
    ax.plot(time, y0, label=z_label)
    ax.plot(time, y1, label=v_label)
    if compare1 is not None:
        ax.plot(time, compare1, label="compare1")
    if compare2 is not None:
        ax.plot(time, compare2, label="compare2")
    ax.legend()
    plt.show()


def test_simpson():
    def f(t: np.ndarray):
        return np.array((np.sin(t), np.cos(t)))

    t = np.array([0, np.pi / 4, np.pi / 2, np.pi * 3 / 4, np.pi])
    y1 = f(t)
    i1 = simpson(y1, x=t)
    assert isinstance(i1, (list, np.ndarray)) and abs(i1[0] - 2.0) < 1e-2, f"Found {i1}"
    assert isinstance(i1, (list, np.ndarray)) and abs(i1[1]) < 1e-12, f"Found {i1}"
    # print("Integrated:", i1)


def test_exp_decay():
    def exponential_decay(t: float, y: np.ndarray):
        return -0.5 * y

    sol = solve_ivp(fun=exponential_decay, t_span=[0, 100], y0=[2], method="DOP853")  # RK45 100x less accurate
    for i, t in enumerate(sol.t):
        assert abs(sol.y[0][i] - 2 * np.exp(-0.5 * t)) < 1e-5, f"Time {t}, {sol.y[0][i]} != {2 * np.exp(-0.5 * t)}"

    # stepwise integration, re-starting at previous end.
    y0 = 2
    for te in range(0, 100, 10):
        sol = solve_ivp(
            fun=exponential_decay, t_span=[te, te + 10], y0=[y0], method="DOP853"
        )  # RK45 100x less accurate
        for i, t in enumerate(sol.t):
            assert abs(sol.y[0][i] - 2 * np.exp(-0.5 * t)) < 1e-5, f"Time {t}, {sol.y[0][i]} != {2 * np.exp(-0.5 * t)}"
        y0 = sol.y[0][-1]


def test_oscillator(show: bool):
    def osc(t: float, y: np.ndarray, b: float = 0.5, w0: float = 1.0, wf: float = 1.0):
        return np.array((-2 * b * y[0] - w0**2 * y[1] + np.sin(wf * t), y[0]), float)

    sol = solve_ivp(fun=osc, t_span=[0, 10], y0=[0.0, 1.0], atol=1e-5)  # , method='DOP853') #RK45 100x less accurate
    print(sol.t, sol.y[0], sol.y[1])
    # stepwise integration, re-starting at previous end.
    y = np.array((0.0, 1.0), float)
    time: list[float] = []
    res: list[list[float]] = [[], []]
    for te in np.linspace(0, 10, 100):
        sol = solve_ivp(osc, t_span=[te, te + 0.1], y0=y, method="DOP853")  # RK45 100x less accurate
        time.extend(sol.t)
        res[0].extend(sol.y[0])
        res[1].extend(sol.y[1])
        y = np.array((sol.y[0][-1], sol.y[1][-1]), float)
    if show:
        do_show(time, res[0], res[1])


def test_ivp(show: bool = False):
    """Perform a few tests to get more acquainted with the IVP solver. Taken from scipy documentation"""

    def upward_cannon(t: float, y: np.ndarray):  # return speed and accelleration as function of (position, speed)
        return [y[1], -9.81]

    def hit_ground(t: np.ndarray, y: np.ndarray):
        return y[0]

    sol = solve_ivp(
        upward_cannon,  # initial value function
        [0, 100],  # time range
        [0, 200],  # start values (position, speed)
        t_eval=[t for t in range(100)],  # evaluate at these points (not only last time value. For plotting)
    )
    assert sol.status == 0, "No events involved. Successful status should be 0"
    assert len(sol.y) == 2, "y is a double vector of (position, speed), which is also reflected in results"
    if show:
        do_show(list(sol.t), sol.y[0], sol.y[1], z_label="pos", v_label="speed")
    # include hit_ground event. Monkey patching function (which mypy, pyright do not like)
    hit_ground.terminal = True  # type: ignore[attr-defined]
    hit_ground.direction = -1  # type: ignore[attr-defined]
    sol = solve_ivp(
        upward_cannon,
        (0, 100),
        np.array((0, 200), float),
        t_eval=np.array([t for t in range(100)]),
        events=hit_ground # type: ignore
    )  
    assert np.allclose(sol.t_events, [2 * 200 / 9.81]), "Time when hitting the ground"  # type: ignore ## it works
    assert np.allclose(sol.y_events, [[0.0, -200.0]]), "Position and speed when hitting the ground"  # type: ignore
    if show:
        do_show(list(sol.t), sol.y[0], sol.y[1], z_label="pos", v_label="speed")


if __name__ == "__main__":
    retcode = pytest.main(["-rA", "-v", __file__, "--show", "True"])
    assert retcode == 0, f"Non-zero return code {retcode}"
    # test_simpson()
    # test_exp_decay()
    # test_oscillator(show=True)
    # test_ivp()
