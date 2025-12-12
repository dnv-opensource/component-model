# ruff: noqa: I001
from collections.abc import Iterable
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import pytest
from fmpy.simulation import simulate_fmu
from fmpy.util import fmu_info, plot_result
from fmpy.validation import validate_fmu
from component_model.model import Model


@pytest.fixture(scope="module")
def oscillator_6d_fmu():
    return _oscillator_6d_fmu()


def _oscillator_6d_fmu():
    """Make FMU and return .fmu file with path."""
    src = Path(__file__).parent.parent / "examples" / "oscillator_6dof_fmu.py"
    assert src.exists(), f"Model file {src} not found."
    fmu_path = Model.build(
        script=str(src),
        dest=Path(__file__).parent.parent / "examples" / "HarmonicOscillator6D.fmu",
        newargs={
            "k": ("1N/m", "1N/m", "1N/m", "1N*m/rad", "1N*m/rad", "1N*m/rad"),
            "c": ("0.1N*s/m", "0.1N*s/m", "0.1N*s/m", "0.1N*m*s/rad", "0.1N*m*s/rad", "0.1N*m*s/rad"),
            "m": "1.0kg",
            "mi": ("1.0 kg*m**2", "1.0 kg*m**2", "1.0 kg*m**2"),
            "x0": ("0.0m", "0.0m", "0.0m", "0.0rad", "0.0rad", "0.0rad"),
            "v0": ("0.0m/s", "0.0m/s", "0.0m/s", "0.0rad/s", "0.0rad/s", "0.0rad/s"),
        },
    )
    return fmu_path


@pytest.fixture(scope="module")
def driver_6d_fmu():
    return _driver_6d_fmu()


def _driver_6d_fmu():
    """Make FMU and return .fmu file with path."""
    src = Path(__file__).parent.parent / "examples" / "driving_force_fmu.py"
    assert src.exists(), f"Model file {src} not found."
    fmu_path = Model.build(
        script=str(src),
        dest=Path(__file__).parent.parent / "examples" / "DrivingForce6D.fmu",
        newargs={"ampl": ("1.0N", "1.0N", "1.0N", "1.0N*m", "1.0N*m", "1.0N*m"), "freq": ("1.0Hz",) * 6},
    )
    return fmu_path


def arrays_equal(
    res: Iterable[Any],
    expected: Iterable[Any],
    eps: float = 1e-7,
):
    len_res = len(list(res))
    len_exp = len(list(expected))
    if len_res != len_exp:
        raise ValueError(f"Arrays of different lengths cannot be equal. Found {len_res} != {len_exp}")
    for i, (x, y) in enumerate(zip(res, expected, strict=False)):
        assert abs(x - y) < eps, f"Element {i} not nearly equal in {x}, {y}"


def do_show(
    traces: dict[str, tuple[list[float], list[float]]],
    xlabel: str = "frequency in Hz",
    ylabel: str = "position/angle",
    title: str = "External force frequency sweep with time between co-sim calls = 1.0",
):
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for label, trace in traces.items():
        _ = ax.plot(trace[0], trace[1], label=label)
    _ = ax.legend()
    plt.show()


# def force(t: float, ampl: float = 1.0, omega: float = 0.1):
#     return np.array((0, 0, ampl * np.sin(omega * t)), dtype=float)


def test_make_fmus(
    oscillator_6d_fmu: Path,
    driver_6d_fmu: Path,
):
    info = fmu_info(filename=str(oscillator_6d_fmu))  # this is a formatted string. Not easy to check
    print(f"Info Oscillator: {info}")
    val = validate_fmu(filename=str(oscillator_6d_fmu))
    assert not len(val), f"Validation of of {oscillator_6d_fmu.name} was not successful. Errors: {val}"

    info = fmu_info(filename=str(driver_6d_fmu))  # this is a formatted string. Not easy to check
    print(f"Info Driver: {info}")
    val = validate_fmu(filename=str(driver_6d_fmu))
    assert not len(val), f"Validation of of {driver_6d_fmu.name} was not successful. Errors: {val}"


# def test_make_system_structure(system_structure: Path):
#     assert Path(system_structure).exists(), "System structure not created"
#     el = read_xml(Path(system_structure))
#     assert isinstance(el, ET.Element), f"ElementTree element expected. Found {el}"
#     ns = el.tag.split("{")[1].split("}")[0]
#     print("NS", ns, system_structure)
#     for s in el.findall(".//{*}Simulator"):
#         assert (Path(system_structure).parent / s.get("source", "??")).exists(), f"Component {s.get('name')} not found"
#     for _con in el.findall(".//{*}VariableConnection"):
#         for c in _con:
#             assert c.attrib in ({"simulator": "drv", "name": "f[2]"}, {"simulator": "osc", "name": "f[2]"})
#


pytest.mark.skip()


def test_use_fmu(oscillator_6d_fmu: Path, show: bool = False):  # , driver_6d_fmu: Path, show: bool = False):
    """Test single FMUs."""
    # sourcery skip: move-assign
    result = simulate_fmu(
        oscillator_6d_fmu,
        stop_time=50,
        step_size=0.01,
        validate=True,
        solver="Euler",
        debug_logging=True,
        logger=print,  # fmi_call_logger=print,
        start_values={"x[2]": 1.0, "c[2]": 0.1},
    )
    if show:
        plot_result(result)


if __name__ == "__main__":
    retcode = pytest.main(args=["-rA", "-v", __file__, "--show", "True"])
    assert retcode == 0, f"Non-zero return code {retcode}"
    import os

    os.chdir(Path(__file__).parent.absolute() / "test_working_directory")
    drv = _driver_6d_fmu()
    osc = _oscillator_6d_fmu()
    # test_make_fmus(osc, drv)
    # test_use_fmu(osc, drv, show=True)
