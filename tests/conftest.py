import logging
import os
from pathlib import Path
from shutil import rmtree

import pytest

from component_model.model import Model


@pytest.fixture(scope="session")
def oscillator_fmu():
    """Make FMU and return .fmu file with path."""
    build_path = Path(__file__).parent.parent / "examples"
    src = Path(__file__).parent.parent / "examples" / "oscillator_fmu.py"
    fmu_path = Model.build(
        script=src,
        dest=build_path,
    )
    return fmu_path


@pytest.fixture(scope="session")
def driver_fmu():
    """Make FMU and return .fmu file with path."""
    build_path = Path(__file__).parent.parent / "examples"
    src = Path(__file__).parent.parent / "examples" / "driving_force_fmu.py"
    fmu_path = Model.build(
        script=src,
        dest=build_path,
        newargs={"ampl": ("3N", "2N", "1N"), "freq": ("3Hz", "2Hz", "1Hz")},
    )
    return fmu_path


@pytest.fixture(scope="package", autouse=True)
def chdir() -> None:
    """
    Fixture that changes the current working directory to the 'test_working_directory' folder.
    This fixture is automatically used for the entire package.
    """
    os.chdir(Path(__file__).parent.absolute() / "test_working_directory")


@pytest.fixture(scope="package", autouse=True)
def test_dir() -> Path:
    """
    Fixture that returns the absolute path of the directory containing the current file.
    This fixture is automatically used for the entire package.
    """
    return Path(__file__).parent.absolute()


output_dirs = [
    "results",
]
output_files = [
    "*test*.pdf",
]


@pytest.fixture(autouse=True)
def default_setup_and_teardown():
    """
    Fixture that performs setup and teardown actions before and after each test function.
    It removes the output directories and files specified in 'output_dirs' and 'output_files' lists.
    """
    _remove_output_dirs_and_files()
    yield
    _remove_output_dirs_and_files()


def _remove_output_dirs_and_files() -> None:
    """
    Helper function that removes the output directories and files specified in 'output_dirs' and 'output_files' lists.
    """
    for folder in output_dirs:
        rmtree(folder, ignore_errors=True)
    for pattern in output_files:
        for file in Path.cwd().glob(pattern):
            _file = Path(file)
            _file.unlink(missing_ok=True)


@pytest.fixture(autouse=True)
def setup_logging(caplog: pytest.LogCaptureFixture) -> None:
    """
    Fixture that sets up logging for each test function.
    It sets the log level to 'INFO' and clears the log capture.
    """
    caplog.set_level("INFO")
    caplog.clear()


@pytest.fixture(autouse=True)
def logger() -> logging.Logger:
    """Fixture that returns the logger object."""
    return logging.getLogger()


def pytest_addoption(parser):
    parser.addoption("--show", action="store", default=False)


@pytest.fixture(scope="session")
def show(request):
    return request.config.getoption("--show") == "False"
