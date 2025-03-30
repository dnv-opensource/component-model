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
                return {
                    "Real": float,
                    "Integer": int,
                    "Boolean": bool,
                    "String": str,
                    "Enumeration": Enum,
                }.get(val, val)


def read_xml(xml: Path | str, sub: str = "modelDescription.xml") -> ET.Element:
    """Read xml file and return `sub` as Element object.

    xml can be

    * a zip file containing the xml file as `sub`
    * a xml file (e.g. modelDescription.xml).
    * a xml literal string. `sub` ignored in this case.
    """
    path = Path(xml)
    el = None
    if path.exists():  # we have a zip file or an xml file
        if is_zipfile(path):
            assert len(sub), "Information on file within zip needed"
            try:
                with ZipFile(path) as zp:
                    xml_string = zp.read(sub).decode()
            except Exception:
                raise BadZipFile(f"Not able to read zip file {xml} or {sub} not found in zipfile") from None
            el = ET.fromstring(xml_string)
        else:
            try:
                el = ET.parse(path).getroot()  # try to read the file directly, assuming a modelDescription.xml file
            except Exception:
                raise AssertionError(f"Could not parse xml file {path}") from None
    elif Path(path, sub).exists():  # unzipped xml path was provided
        try:
            el = ET.parse(Path(path, sub)).getroot()
        except ET.ParseError:
            raise AssertionError(f"Could not parse xml file {Path(path, sub)}") from None
    elif isinstance(xml, str):
        try:
            el = ET.fromstring(xml)  # try as literal string
        except ET.ParseError as err:
            raise AssertionError(
                f"Error when parsing {xml} as xml file. Error code {err.code} at {err.position}"
            ) from err
    else:
        raise Exception(f"Not possible to read model description from {xml}, {sub}") from None
    assert el is not None, f"xml {xml} not found or {sub} could not be read"
    return el
