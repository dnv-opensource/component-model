import xml.etree.ElementTree as ET
from enum import Enum
from pathlib import Path
from zipfile import BadZipFile, ZipFile, is_zipfile


def xml_to_python_val(val: str):
    """Translate the xml (string) value to a python value and type."""
    if val == "true":
        return True
    if val == "false":
        return False
    try:
        return int(val)
    except Exception:  # noqa: BLE001
        try:
            return float(val)
        except Exception:  # noqa: BLE001
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
            except Exception as e:
                raise BadZipFile(f"Not able to read zip file {xml} or {sub} not found in zipfile") from e
            el = ET.fromstring(xml_string)  # noqa: S314
        else:
            try:
                # try to read the file directly, assuming a modelDescription.xml file
                el = ET.parse(path).getroot()  # noqa: S314
            except Exception as e:
                raise RuntimeError(f"Could not parse xml file {path}") from e
    elif Path(path, sub).exists():  # unzipped xml path was provided
        try:
            el = ET.parse(Path(path, sub)).getroot()  # noqa: S314
        except ET.ParseError as e:
            raise RuntimeError(f"Could not parse xml file {Path(path, sub)}") from e
    elif isinstance(xml, str):
        try:
            # try as literal string
            el = ET.fromstring(xml)  # noqa: S314
        except ET.ParseError as e:
            raise RuntimeError(f"Error when parsing {xml} as xml file. Error code {e.code} at {e.position}") from e
    else:
        raise RuntimeError(f"Not possible to read model description from {xml}, {sub}")
    assert el is not None, f"xml {xml} not found or {sub} could not be read"
    return el
