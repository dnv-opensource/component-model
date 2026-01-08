# ruff: noqa: I001
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
from fmpy.simulation import simulate_fmu  # type: ignore[import-untyped]
from fmpy.util import fmu_info, plot_result  # type: ignore[import-untyped]
from fmpy.validation import validate_fmu  # type: ignore[import-untyped]

from component_model.model import Model


@pytest.fixture(scope="module")
def oscillator_fmu():
    return _oscillator_fmu()


def _oscillator_fmu():
    """Make FMU and return .fmu file with path."""
    fmu_path = Model.build(
        script=Path(__file__).parent.parent / "examples" / "oscillator_fmu.py",
        dest=Path(__file__).parent.parent / "examples" / "HarmonicOscillator.fmu",
    )
    return fmu_path


@pytest.fixture(scope="module")
def driver_fmu():
    return _driver_fmu()


def _driver_fmu():
    """Make FMU and return .fmu file with path."""
    fmu_path = Model.build(
        script=Path(__file__).parent.parent / "examples" / "driving_force_fmu.py",
        dest=Path(__file__).parent.parent / "examples" / "DrivingForce.fmu",
        newargs={"ampl": ("3N", "2N", "1N"), "freq": ("3Hz", "2Hz", "1Hz")},
    )
    return fmu_path


def do_show(traces: dict[str, tuple[list[float], list[float]]]):
    fig, ax = plt.subplots()
    for label, trace in traces.items():
        _ = ax.plot(trace[0], trace[1], label=label)
    _ = ax.legend()
    plt.show()


# def force(t: float, ampl: float = 1.0, omega: float = 0.1):
#     return np.array((0, 0, ampl * np.sin(omega * t)), dtype=float)


@pytest.fixture(scope="session")
def system_structure():
    return _system_structure()


def _system_structure():
    """Make a OSP structure file and return the path"""
    return Path(__file__).parent.parent / "examples" / "ForcedOscillator.xml"


def test_make_fmus(
    oscillator_fmu: Path,
    driver_fmu: Path,
    show: bool = False,
):
    info = fmu_info(filename=str(oscillator_fmu))  # this is a formatted string. Not easy to check
    if show:
        print(f"Info Oscillator @{oscillator_fmu}")
        print(info)
    val = validate_fmu(filename=str(oscillator_fmu))
    assert not len(val), f"Validation of of {oscillator_fmu.name} was not successful. Errors: {val}"

    info = fmu_info(filename=str(driver_fmu))  # this is a formatted string. Not easy to check
    if show:
        print(f"Info Driver: @{driver_fmu}")
        print(info)
    val = validate_fmu(filename=str(driver_fmu))
    assert not len(val), f"Validation of of {driver_fmu.name} was not successful. Errors: {val}"


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


def test_run_fmpy(oscillator_fmu: Path, driver_fmu: Path, show: bool = False):
    """Test single FMUs."""
    # sourcery skip: move-assign
    result = simulate_fmu(
        oscillator_fmu,
        stop_time=50,
        step_size=0.01,
        validate=True,
        solver="Euler",
        debug_logging=True,
        logger=print,  # fmi_call_logger=print,
        start_values={"x[2]": 1.0, "c[2]": 0.1},
        step_finished=None,  # pyright: ignore[reportArgumentType]  # (typing incorrect in fmpy)
        fmu_instance=None,  # pyright: ignore[reportArgumentType]  # (typing incorrect in fmpy)
    )
    if show:
        plot_result(result)


def test_run_fmpy2(oscillator_fmu: Path, driver_fmu: Path, show: bool = False):
    """Test oscillator in setting similar to 'crane_on_spring'"""
    # sourcery skip: move-assign
    result = simulate_fmu(
        oscillator_fmu,
        stop_time=10,
        step_size=0.01,
        validate=True,
        solver="Euler",
        debug_logging=True,
        logger=print,  # fmi_call_logger=print,
        start_values={
            "m": 10000.0,
            "k[0]": 10000.0,
            "k[1]": 10000.0,
            "k[2]": 10000.0,
            "x[0]": 0.0,
            "x[1]": 0.0,
            "x[2]": 0.0,
            "v[0]": 1.0,
        },
        step_finished=None,  # pyright: ignore[reportArgumentType]  # (typing incorrect in fmpy)
        fmu_instance=None,  # pyright: ignore[reportArgumentType]  # (typing incorrect in fmpy)
    )
    if show:
        plot_result(result)


if __name__ == "__main__":
    retcode = pytest.main(args=["-rA", "-v", __file__, "--show", "True"])
    assert retcode == 0, f"Non-zero return code {retcode}"
    import os

    os.chdir(Path(__file__).parent.absolute() / "test_working_directory")
    osc = _oscillator_fmu()
    drv = _driver_fmu()
    # test_make_fmus(osc, drv, show=True)
    # test_run_fmpy(osc, drv, show=True)
    # test_run_fmpy2(osc, drv, show=True)
