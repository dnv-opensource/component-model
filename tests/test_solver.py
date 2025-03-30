import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate


def do_show(time: list, y0: list, y1: list):
    fig, ax = plt.subplots()
    ax.plot(time, y0, label="position")
    ax.plot(time, y1, label="speed")
    ax.legend()
    plt.show()


def test_simpson():
    def f(t):
        return np.array((np.sin(t), np.cos(t)))

    t = np.array([0, np.pi / 4, np.pi / 2, np.pi * 3 / 4, np.pi])
    y1 = f(t)
    i1 = integrate.simpson(y1, x=t)
    assert isinstance(i1, (list, np.ndarray)) and abs(i1[0] - 2.0) < 1e-2, f"Found {i1}"
    assert isinstance(i1, (list, np.ndarray)) and abs(i1[1]) < 1e-12, f"Found {i1}"
    # print("Integrated:", i1)


def test_exp_decay():
    def exponential_decay(t, y):
        return -0.5 * y

    sol = integrate.solve_ivp(
        fun=exponential_decay, t_span=[0, 100], y0=[2], method="DOP853"
    )  # RK45 100x less accurate
    for i, t in enumerate(sol.t):
        assert abs(sol.y[0][i] - 2 * np.exp(-0.5 * t)) < 1e-5, f"Time {t}, {sol.y[0][i]} != {2 * np.exp(-0.5 * t)}"

    # stepwise integration, re-starting at previous end.
    y0 = 2
    for te in range(0, 100, 10):
        sol = integrate.solve_ivp(
            fun=exponential_decay, t_span=[te, te + 10], y0=[y0], method="DOP853"
        )  # RK45 100x less accurate
        for i, t in enumerate(sol.t):
            assert abs(sol.y[0][i] - 2 * np.exp(-0.5 * t)) < 1e-5, f"Time {t}, {sol.y[0][i]} != {2 * np.exp(-0.5 * t)}"
        y0 = sol.y[0][-1]


def test_oscillator(show):
    def osc(t, y, b=0.5, w0=1.0, wf=1.0):
        return np.array((-2 * b * y[0] - w0**2 * y[1] + np.sin(wf * t), y[0]), float)

    sol = integrate.solve_ivp(
        fun=osc, t_span=[0, 10], y0=[0.0, 1.0], atol=1e-5
    )  # , method='DOP853') #RK45 100x less accurate
    print(sol.t, sol.y[0], sol.y[1])
    # stepwise integration, re-starting at previous end.
    y = [0.0, 1.0]
    time = []
    res: list[list] = [[], []]
    for te in np.linspace(0, 10, 100):
        sol = integrate.solve_ivp(osc, t_span=[te, te + 0.1], y0=y, method="DOP853")  # RK45 100x less accurate
        time.extend(sol.t)
        res[0].extend(sol.y[0])
        res[1].extend(sol.y[1])
        y = [sol.y[0][-1], sol.y[1][-1]]
    if show:
        do_show(time, res[0], res[1])


if __name__ == "__main__":
    retcode = 0  # pytest.main(["-rA", "-v", __file__, "--show", "True"])
    assert retcode == 0, f"Non-zero return code {retcode}"
    test_simpson()
    # test_exp_decay()
    # test_oscillator(show=True)
