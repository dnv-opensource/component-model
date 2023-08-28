from abc import ABC, abstractmethod
from pint import UnitRegistry
from .variable import Variable
from math import log
import xml.etree.ElementTree as ET


class ModelInitError(Exception):
    '''Special error indicating that something is wrong with the boom definition'''
    pass
class ModelOperationError(Exception):
    '''Special error indicating that something went wrong during crane operation (rotations, translations,calculation of CoM,...)'''
    pass
class ModelAnimationError(Exception):
    '''Special error indicating that something went wrong during crane animation'''
    pass


class Model:
    """ Defines an abstract model including some common model concepts, like variables and units.
    The model interface and the inner working of the model is missing here and must be defined by derived classes.
    For a fully defined derived model class it shall be possible to
    
    * define a full set of interface variables
    * set the current variable values
    * run the model in isolation for a time interval
    * retrieve updated variable values
    
    The following FMI concepts are (so far) not implemented
    
    * TypeDefinitions. Instead of defining SimpleType variables, ScalarVariable variables are always based on the pre-defined types and details provided there
    * DisplayUnit. Variable units contain a Unit (the unit as used for inputs and outputs) and BaseUnit (the unit as used in internal model calculations, i.e. based on SI units).
      Additional DisplayUnit(s) are so far not defined/used. Unit is used for that purpose.
    
    Args:
        name (str): unique model of the instantiated model
        author (str) = 'anonymous': The author of the model
        version (str) = '0.1': The version number of the model
        unitSystem (str)='SI': The unit system to be used. self.uReg.default_system contains this information for all variables
    """
    def __init__(self, name, description:str='A component model', author:str='anonymous', version:str='0.1', unitSystem='SI'):
        self.name = name
        self.description = description
        self.author = author
        self.version = version
        self._variables = [] # keep track of all instantiated variables
        self.varRefCount = 0 # since variables can have several components, we need to keep a variable refCount separate to the variables list
        self.uReg = UnitRegistry( system=unitSystem, autoconvert_offset_to_baseunit=True) # use a common UnitRegistry for all variables
        self._units = [] # list of the units used in the model (for usage in UnitDefinitions element)
        self.onModelFreeze = [] # list of functions which is run when model is frozen
        self.currentTime = 0 # keeping track of time when dynamic calculations are performed
        self.changedVariables = [] # list of input variables which are changed at any time, i.e. the variables which are taken into account during do_step()
                                   # A changed variable is kept in the list until its value is not changed any more
                                   # Adding/removing of variables happens through change_variable()
        self._eventList = [] # possibility for a list of events that will be activated on time during a simulation
                             # Events consist of tuples of (time, changedVariable)


    @property
    def variables(self): return(self._variables)
    def variables_append(self, var):
        '''Register the variable 'var' as model variable. Add the unit if not yet used. Perform some checks and return the index (useable as valueReference)
        Note that the variable name, _initialVal and _unit must be set before calling this function 
        '''
        for v in self._variables:
            if v.name == var.name:
                raise ModelInitError("Variable name " +var.name +" is not unique")
        self._variables.append( var) # register the variable
        if var.displayUnit not in self._units:
            self._units.append( var.displayUnit) # keep track of the various units used in the model
        idx = self.varRefCount
        if isinstance( var.initialVal, tuple): self.varRefCount += len( var.initialVal) # advance the counter so that there are enough unique valueReference indices
        else:                                  self.varRefCount += 1
        return( idx)
    
    @property
    def units(self): return(self._units)
    
    def add_variable(self, *args, **kwargs):
        '''Convenience method, where the model reference is automatically added to the variable initialization'''
        return( Variable( self, *args, **kwargs))
    
    def add_event(self, time:float|None=None, event:tuple=None):
        '''Register a new event to the event list. Ensure that the list is sorted.
        Note that the event mechanism is mainly used for model testing, since normally events are initiated by input variable changes.

        Args:
            time (float): the time at which the event shall be issued. If None, the event shall happen immediatelly
            event (tuple): tuple of the variable (by name or object) and its changed value
        '''
        if event is None: return # no action
        var = event[0] if event[0] in self._variables else self.variable_by_name( event[0])
        if var is None:
            raise ModelOperationError("Trying to add event related to unknown variable " +str( event[0]) +". Ignored.")
            return        
        if time is None:
            self._eventList.append(-1, (var, event[1])) # append (the list is sorted wrt. decending time)            
        else:
            if not len( self._eventList):
                self._eventList.append( ( time, (var, event[1])))
            else:
                for i, (t,_) in enumerate( self._eventList):
                    if t<time:
                        self._eventList.insert( i, (time,( var, event[1])))
                        break
                    
    def variable_by_name(self, name:str, errorMsg:str|None=None):
        '''Return Variable object related to name, or None, if not found.
        If errorMsg is not None, an error is raised and the message provided '''
        for var in self._variables:
            if var.name == name:
                return( var)
        if errorMsg is not None:
            raise ModelInitError(errorMsg)
        return( None)

    def make_unit_definitions(self):
        '''Make the xml element for the unit definitions used in the model. See FMI 2.0.4 specification 2.2.2'''
        unitDefinitions = ET.Element('UnitDefinitions')
        for u in self._units:
            uBase = self.uReg(u).to_base_units()
            dim = uBase.dimensionality
            exponents = {}
            for key,value in { 'mass':'kg', 'length':'m', 'time':'s', 'current':'A', 'temperature':'K', 'substance':'mol', 'luminosity':'cd'}.items():
                if '['+key+']' in dim:
                    exponents.update({ value : str(int(dim['['+key+']']))})
            print(uBase, uBase.units)
            if 'radian' in str(uBase.units): # radians are formally a dimensionless quantity. To include 'rad' as specified in FMI standard this dirty trick is used
                uDeg = str(uBase.units).replace('radian','degree')
                print("EXPONENT", uBase.units, uDeg, log(uBase.magnitude), log(self.uReg('degree').to_base_units().magnitude))
                exponents.update( {'rad':str(int( log(uBase.magnitude) / log(self.uReg('degree').to_base_units().magnitude)))})

            unit = ET.Element("Unit", {'name':u})
            base = ET.Element("BaseUnit", exponents)
            base.attrib.update( {'factor':str(self.uReg(u).to_base_units().magnitude)})
            unit.append( base)
            unitDefinitions.append( unit)
        return( unitDefinitions)
        