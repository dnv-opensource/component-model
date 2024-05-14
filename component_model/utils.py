import xml.etree.ElementTree as ET  # noqa: N817
from enum import Enum
from pathlib import Path
from zipfile import BadZipFile, ZipFile, is_zipfile


def xml_to_python_val(val: str):
    """Translate the xml (string) value to a python value and type."""
    if val == "true":
        return True
    elif val == "false":
        return False
    else:
        try:
            return int(val)
        except Exception:
            try:
                return float(val)
            except Exception:
                if val == "Real":
                    return float
                elif val == "Integer":
                    return int
                elif val == "Boolean":
                    return bool
                elif val == "String":
                    return str
                elif val == "Enumeration":
                    return Enum
                else:
                    return val  # seems to be a free string


def read_model_description(fmu: Path | str, sub: str = "modelDescription.xml") -> ET.Element:
    """Read FMU file and return as Element object.
    fmu can be the full FMU zipfile, the modelDescription.xml or a equivalent string.
    """
    path = Path(fmu)
    el = None
    if path.exists():  # we have a zip file or an xml file
        if is_zipfile(path):
            assert len(sub), "Information on file within zip needed"
            try:
                with ZipFile(path) as zp:
                    fmu_string = zp.read(sub).decode()
            except Exception:
                raise BadZipFile(f"Not able to read zip file {fmu} or {sub} not found in zipfile") from None
            el = ET.fromstring(fmu_string)
        else:
            try:
                el = ET.parse(path).getroot()  # try to read the file directly, assuming a modelDescription.xml file
            except Exception:
                raise AssertionError(f"Could not parse xml file {path}") from None
    elif Path(path, sub).exists():  # unzipped fmu path was provided
        try:
            el = ET.parse(Path(path, sub)).getroot()
        except ET.ParseError:
            raise AssertionError(f"Could not parse xml file {Path(path,sub)}") from None
    elif isinstance(fmu, str):
        try:
            el = ET.fromstring(fmu)  # try as literal string
        except ET.ParseError as err:
            raise AssertionError(
                f"Error when parsing {fmu} as xml file. Error code {err.code} at {err.position}"
            ) from err
    else:
        raise Exception(f"Not possible to read model description from {fmu}, {sub}") from None
    assert el is not None, f"FMU {fmu} not found or {sub} could not be read"
    return el
