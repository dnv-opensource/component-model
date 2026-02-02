import logging
import os
import xml.etree.ElementTree as ET  # noqa: N817
from pathlib import Path
from shutil import rmtree
from typing import Any

import pytest

from component_model.utils.xml import read_xml


def system_structure_change(structure_file: Path, change: dict[str, tuple[str, str]], newname: str | None = None):
    """Do changes to an existing 'structure_file' and save as newname.
    Changes are provided as dict: { tag : (what, newval),...}
    where 'what'='text' marks the text part else, an attribute is assumed.
    """

    def register_all_namespaces(filename: Path):
        namespaces: dict[Any, Any] = {}
        for _, (ns, uri) in ET.iterparse(filename, events=["start-ns"]):
            # print("TYPES", ns, type(ns), uri, type(uri))
            namespaces.update({ns: uri})
            ET.register_namespace(str(ns), str(uri))
        #         namespaces: dict = dict([node )])
        #        for ns in namespaces:
        #            ET.register_namespace(ns, namespaces[ns])
        return namespaces

    if newname is None:
        newname = structure_file.name  # same as old
    nss = register_all_namespaces(structure_file)
    el = read_xml(structure_file)
    for tag, (what, newval) in change.items():
        elements = el.findall(f".//ns:{tag}", {"ns": nss[""]})
        for e in elements:
            if what == "text":
                e.text = newval
            else:  # assume attribute name
                e.attrib[what] = newval
    path = structure_file.parent / newname
    ET.ElementTree(el).write(path, encoding="utf-8")
    return path


# @pytest.fixture(scope="session", autouse=True)
# def instantiate_cosim_execution() -> None:
#     """
#     Fixture that instantiates a CosimExecution object for the entire package.
#     This fixture is automatically used for the entire package.
#     """
#
#     from libcosimpy.CosimExecution import CosimExecution
#
#     _ = CosimExecution.from_step_size(1)
#     return


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
    "data",
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


def pytest_addoption(parser: pytest.Parser):
    parser.addoption("--show", action="store", default=False)


@pytest.fixture(scope="session")
def show(request: pytest.FixtureRequest):
    return request.config.getoption("--show") == "False"
