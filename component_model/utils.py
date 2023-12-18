from enum import Enum
from pathlib import Path
from zipfile import is_zipfile, ZipFile
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

def read_model_description( fmu:Path):
    """Read FMU file and return as ElementTree object. fmu can be the full FMU zipfile, the modelDescription.xml or a equivalent string"""
    try:
        with ZipFile(fmu) as zp:
            el = ET.fromstring(zp.read("modelDescription.xml"))
    except:
        try:
            el = ET.parse(fmu)
        except:
            try:
                el = ET.parse( Path( fmu, "/modelDescription.xml"))
            except:
                try:
                    el = ET.fromstring( fmu) # assume a literal string
                except: # give up
                    if provideMsg: print( f"The supplied parameter '{fmu}' is neither a zipped FMU, an FMU folder or a modelDescription.xml file")
                    return None
    return el
