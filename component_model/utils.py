from enum import Enum
from pathlib import Path
from zipfile import is_zipfile, ZipFile, BadZipFile
import xml.etree.ElementTree as ET


def xml_to_python_val( val:str):
    '''Translate the xml (string) value to a python value and type'''
    if val=='true': return True
    elif val=='false': return False
    else:
        try:
            return int( val)
        except:
            try:
                return float( val)
            except:
                if val=='Real'         : return(float)
                elif val=='Integer'    : return( int)
                elif val=='Boolean'    : return( bool)
                elif val=='String'     : return( str)
                elif val=='Enumeration': return(Enum)
                else:                    return val  # seems to be a free string

def read_model_description( fmu:Path|str, description_file='modelDescription.xml'):
    """Read FMU file and return as ElementTree object. fmu can be the full FMU zipfile, the modelDescription.xml or a equivalent string"""
    el = fmu_string = None
    if is_zipfile( fmu):
        try:
            with ZipFile(fmu) as zp:
                fmu_string = zp.read( description_file)
        except BadZipFile:
            print(f"Not able to read zip file {fmu}")
    else: 
        try:
            el = ET.parse(fmu)  # try to read the file directly, assuming a modelDescription.xml file
        except:
            try:
                el = ET.parse( Path( fmu, "/modelDescription.xml"))  # maybe the unzipped fmu path was provided?
            except:
                fmu_string = fmu
    if fmu_string is not None and el is None: # have so far only an fmu as string
        try:
            el = ET.fromstring( fmu_string)  # try as literal string
        except ET.ParseError as err:
            print( f"Error when parsing {fmu_string} as xml file. Error code {err.code} at {err.position}")
    return el
