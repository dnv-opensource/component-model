from __future__ import annotations
from . import logger
logger = logger.get_module_logger(__name__, level=0)

from enum import Enum, IntFlag
from math import pi, radians, degrees, sin, cos, sqrt, acos, atan2
import xml.etree.ElementTree as ET
from pythonfmu.variables import ScalarVariable
from pythonfmu.enums import Fmi2Causality as Causality, Fmi2Initial as Initial, Fmi2Variability as Variability
import copy
import numpy as np
from pint import UnitRegistry, Quantity # management of units
from scipy.spatial.transform import Rotation as Rot

# Some special error classes
class VariableInitError(Exception):
    '''Special error indicating that something is wrong with the variable definition'''
    pass
class VariableRangeError(Exception):
    '''Special Exception class signalling that a value is not within the range'''
    pass
class VariableUseError(Exception):
    '''Special Exception class signalling that variable use was not in accordance with settings'''
    pass
class VarCheck(IntFlag):
    '''Flags to denote how variables should be checked with respect to units and range. The aspects are indepent, but can be combined in the Enum through |
    
    * none:       neither units nor ranges are expected or checked.
    * unitNone:   only numbers without units expected when new values are provided.
      If units are provided during initialization, these should be base units (SE), i.e. unit and displayUnit are the same.
    * unitAll:    expect always quantity and number and convert internally to base units (SE). Provide output as displayUnit
    * units:      flag to filter only on units, e.g ck & VarCheck.units
    * rangeNone:  no range is provided or checked
    * rangeCheck: range is provided and checked
    * ranges:     flag to filter on range, e.g. ck & VarCheck.ranges
    * all:        short for unitAll | rangeCheck
    '''
    none       = 0
    unitNone   = 0 
    unitAll    = 1 
    units      = 1 
    rangeNone  = 0 
    rangeCheck = 2 
    ranges     = 2
    all        = 2

class Variable(ScalarVariable):
    '''Interface variable of an FMU. Can be a (python type) scalar variable. Extensions cover arrays (e.g. numpy array).
    The class extends pythonfmu.ScalarVariable, not using the detailed types (Real, Integer, ...), as these are handled internally
    The recommended way to instantiate a Variable is through string values (with units for initialVal and rng),
    but also dimensionless quantities and explicit class types are accepted.
    Note1: _displayUnit is set to None if it is equal to _unit or if dimensionless units are used, as a quick signal that no conversion is needed 
    Note2: The valueReference is automatically managed and cannot be manually set.
    
        Args:
            model (obj): The model object where this variable relates to. Use model.add_variable( name, ...) to define variables
            name (str): Variable name, unique for whole FMU
            description (str) = None: Optional description of variable
            causality (str) = 'parameter': The causality setting as string
            variability (str) = 'fixed': The variability setting as string
            initial (str) = None: Definition how the variable is initialized. Provide this explicitly if the default value is not suitable. 
            canHandleMultipleSetPerTimeInstant (bool) = False: (only used for ModelExchange). Determines when in simulation it is allowed to set the variable, i.e. loops are not allowed when False
            typ (type)=None: The type of variable to expect as initialVal and value. Since initial values are often set with strings (with units, see below), this is set explicitly.
               If None, _type is set to Enum/str if derived from these after disection or float if a number. 'int' is not automatically detected.
            initialVal (str,int,float,Enum): The initial value of the variable.
            
               Optionally, the unit can be included, providing the initial value as string, evaluating to quantity of type typ a display unit and base unit.
               Note that the quantities are always converted to standard units of the same type, while the display unit may be different, i.e. the preferred user communication.
            rng (tuple) = (): Optional range of the variable in terms of a tuple of the same type as initial value. Can be specified with units (as string).
            
               * If an empty tuple is specified, the range is automatically determined. Note that it is thus not possible to automatically set single range elements (lower,upper) automatically
               * If None is specified, the initial value is chosen, i.e. no range. Applies to whole range tuple or to single elements (lower,upper)
               * For derived classes of Variable, the scalar ranges are in general calculated first and then used to specify derived ranges
               * For some variable types (e.g. str) no range is expected.
               
            annotations (dict) = {}: Optional variable annotations provided as dict
            valueCheck (VarCheck) = VarCheck=VarCheck.rangeCheck|VarCheck.unitAll: Setting for checking of units and range according to VarCheck. The two aspects should be set with OR (|)
            fullInit (bool) = True: Optional possibility to stop the initialization of single variables, where this does not make sense for derived, compound variables
            on_step (callable) = None: Optonal possibility to register a function of (currentTime, dT) to be run during Model.do_step,
               e.g. if the variable represents a speed, the object can be translated speed*dT, if |speed|>0
            on_set (callable) = None: Optional possibility to specify a pre-processing function of (newVal) to be run when the variable is initialized or changed.
               This is useful for conditioning of input variables, so that calculations can be done once after a value is changed and do not need to be repeated on every simulation step.
               If given, the function shall apply to a value as expected by the variable (e.g. if there are components) and after unit conversion and range checking.
               The function is completely invisible by the user specifying inputs to the variable.
        
        .. todo:: Warnings on used default values which should be provided explicitly to conform to RP-0513
        .. limitation:: Limitation test 
        .. assumption:: Assumption test
        .. requirement:: Requirement test
    '''
    def __init__(self, model, name: str, description:str=None, causality:str='parameter', variability:str='fixed', initial:str=None, canHandleMultipleSetPerTimeInstant: bool = False,
                 typ:type|None=None, initialVal: str=None, rng: tuple = (),
                 annotations: dict = {}, valueCheck:VarCheck=VarCheck.all, fullInit:bool=True, on_step:callable=None, on_set:callable=None):
        self.model = model
        super().__init__( name=name, description=description, causality=Causality[causality], variability=Variability[variability], initial=initial, getter=self.getter, setter=self.setter) #Note that the variables class does not relate to valued
        _initial = self.check_causality_variability_initial( initial)
        if _initial is None:
            raise VariableInitError( "The combination for causality '" +str(causality) +"' + variability '" +str(variability) +"' is not allowed.")
        else:
            self._ScalarVariable__attrs['initial'] = _initial
        if canHandleMultipleSetPerTimeInstant and self._variability!=Variability.input:
            raise VariableInitError( "Inconsistent 'canHandleMultipleSetPerTimeInstant' detected. Should only be used for 'input' variables")
        self._canHandleMultipleSetPerTimeInstant = canHandleMultipleSetPerTimeInstant
        self._annotations = annotations
        self._valueCheck = valueCheck
        self.on_step = on_step # hook to define a function of currentTime and time step dT, to be performed during Model.do_step for input variables
        self.on_set = on_set # hook to perform initial calculations when an input value is
        self._len = 1
        if getattr( self, "fullInit", True): #this attribute can be set by super-classes to stop the initialisation here
            self._initialVal, self._type, self.unit, self.displayUnit, self._range, self._value = self.initialVal_setter( initialVal, rng, typ)
            self._idx = self.model.register_variable( self) # register in model and return index
            self.valueReference = self._idx # always automatic reference number

    def __len__(self): return( self._len) # This works also for derived variables, as long as _len is properly set
    # some getter and setter methods
    @property
    def type(self): return( self._type)
    @property
    def initialVal(self): return( self._initialVal)
    def initialVal_setter(self, val, rng, _type): # the initialVal is set during instantiation and is not meant to be changed!
        if _type==str: # explicit free string
            return( val, _type, '', None, tuple(), self.on_set( val) if self.on_set is not None else val)
        
        _initialVal, _unit, _displayUnit = self.disect_unit( val) # first disect the initial value
        if _type is None: _type = self.auto_type( _initialVal)
        self._type = _type # make it available for the functions below
        _initialVal = _type(_initialVal)
        _range = self.init_range( rng, _initialVal, _unit) if VarCheck.rangeCheck in self._valueCheck else None
        _value = self.on_set( _initialVal) if self.on_set is not None else _initialVal #.. then pre-process if on_set is defined
        return( _initialVal, _type, _unit, _displayUnit, _range, _value)

#     @property
#     def initial(self):
#         return( self._ScalarVariable__attrs['initial'])
#     @initial.setter
#     def initial(self, newVal):
#         self._ScalarVariable__attrs['initial'] = newVal
    @property
    def range(self): return( self._range)
    
    @property
    def value(self): return( self._value)
    @value.setter
    def value(self, newVal):
        '''Set the variable (if allowed and if range check valid). This method is designed for variable setting within the model.
         For FMU input settings the model object has functions ('Model._set') with additional functionality (e.g. unit handling).
        '''
        if VarCheck.rangeCheck in self._valueCheck: # range checking is performed
            if not self.check_range( newVal, rng=self._range):
                raise VariableRangeError("The value " +str(newVal) +" is not accepted within variable " +self.name +". Range is set to " +str(self._range))
        self._value = self.on_set( newVal) if self.on_set is not None else newVal #.. then pre-process if on_set is defined
    
    @property
    def valueReference(self): return(self._valueReference)
    @valueReference.setter
    def valueReference(self, newRef):
        self._valueReference = newRef
        
    def auto_type(self, exampleVal):
        '''Determine the Variable type from a provided example value.
        Since variables can be initialized using strings with units, the type can only be determined when the value is disected.
        Moreover, the value may indicate an integer, while the variable is designed a float. Therefore int Variables must be explicitly specified.'''
        return( float if isinstance( exampleVal, (int,float)) else type( exampleVal))

    def check_value(self, val:str|int|float|Enum, _initialVal:int|float|Enum|str|None=None, _range:tuple|None=None, _displayUnit:str|None=None):
        '''Checks a provided value and returns the quantity, unit, displayUnit and range. Processing like on_set is not performed here.
        Note: The function works also for components of derived variables if all input parameters are provided explicitly
        Note 2: Variable initialization is not handled here
        .. todo:: better coverage with respect to variability and causality on when it is allowed to change the value.
        
        Args:
            val: the raw value to be checked. May be a string with units (to be disected as part of this function)
            _initialVal (int,float,...)=None: the processed initialVal (unit disected and range checked). self._initialVal if None and not initial
            _range (tuple)=None: the range provided as tuple. self._range if None and not initial
            _displayUnit (str)=None: the displayUnit (unit as which the variable is expected). self.displayUnit if None and not initial
        Returns:
            val
        '''
        if self.variability==Variability.constant:
            raise VariableUseError("It is not allowed to change the value of variable " +self.name)
        if VarCheck.unitAll in self._valueCheck:
            val, _unit, dU = self.disect_unit( val) # first disect the initial value
        if VarCheck.unitAll in self._valueCheck: # units are expected (disected above) and are checked for consistency
            if _displayUnit is None: _displayUnit = self.displayUnit
            if dU != _displayUnit:
                raise VariableUseError("The expected unit of variable " +self.name +" is " +_displayUnit +". Got " +dU)    
        if VarCheck.rangeCheck in self._valueCheck: # range checking is performed
            if _initialVal is None: _initialVal = self._initialVal
            if _range is None: _range = self._range
            if not self.check_range( val, rng=_range):
                raise VariableRangeError("The value " +str(val) +" is not accepted within variable " +self.name +". Range is set to " +str(_range))
        return( val)

    def __str__(self):
        return("Variable " +self.name +". Initial: " +str(self._initialVal) +". Current " +str(self._value) +" ["+str(self.unit)+"]")

    @classmethod
    def _get_auto_extreme(self, var):
        '''Return the extreme value of the variable
        
        Args:
            var: the variable for which to determine the extremes. Represented by an instantiated object
        Returns:
            A tuple containing the minimum and maximum value the given variable can have
        '''
        if isinstance(var, float):  return( ( float('-inf'), float('inf'))) 
        elif isinstance(var, int):  raise VariableInitError(f"Unlimited integer variables do not make sense in Python. Please provide explicit limits for variable {self.name} or set the type to float.")
        elif isinstance(var, Enum): return( min( x.value for x in type(var)), max( x.value for x in type(var)))
        else:                       return( tuple()) # return an empty tuple (no range specified, e.g. for str)

    def init_range(self, rng:tuple, initialVal=None, unit:str=None):
        '''Initialize the variable, including unit and range.
        Function can be called separately per component by derived classes
        The initialVal and unit can be explicitly provided, or self._* is used, if None'''
        if initialVal is None: initialVal = self._initialVal
        if unit is None: unit = self.unit
        #=== initialize the variable range
        if rng is None: # set a zero-interval range. Note: this makes only sense for combined variables, where some components may be fixed and others variable
            _range = (initialVal, initialVal)
        elif not len( rng): # empty tuple => automatic range (for float and Enum)
            _range = self._get_auto_extreme( initialVal)
        elif len( rng) == 1:
            raise VariableInitError( f"Range specification of variable {self.name} is unclear, because only one value is provided")
        elif len( rng)==2:
            _rng = []
            for i,r in enumerate(rng):
                if r is None: # no range => fixed to initial value
                    q = initialVal
                else:
                    q,u,dU = self.disect_unit( r)
                    if (q==0 or q==float('inf') or q==float('-inf')) and u=='dimensionless': # we accept that no explicit unit is supplied when the quantity is 0 or inf
                        u = unit
                    elif len(unit) and len(u) and unit != u:
                        raise VariableInitError("The supplied range value " +str(r) +" does not conform to the unit type " +unit)
                _rng.append( q) 
            try: # check variable type
                _rng = [ type(initialVal)( x) for x in _rng]
            except:
                raise VariableRangeError("The given range " +str(rng) +" is not compatible with the provided value " +str( initialVal))
            if all( isinstance( x, type( initialVal)) for x in _rng):
                _range = tuple( _rng)
            else:
                raise NotImplementedError("What else?")
        else:
            raise VariableRangeError("Something wrong with this range specification (length): " +str( rng))
        if not self.check_range( initialVal, _range): # check also whether the provided (initial) _initialValue is within the determined range
            raise VariableInitError("The provided value " +str(initialVal) +" is not in the valid range " +str(_range))
        return( _range)
    
    def check_range(self, val: int|float|str|Enum, rng:tuple|None=None) -> bool:
        '''Check the provided 'val' with respect to the variable range

        Args:
            val (int,float,str,Enum): the value to check.
            
               * If tuple, check whether the provided range tuple is adequate with respect to variable type.
               * It is allowed to replace a value within the tuple by None, but only one value is not allowed
               * If None, the range is set equal to the initalVal, which makes only sense if the Variable is a component of a derived Variable, where other components are not constants
               * If not tuple or None, check with respect to variable type and whether it is within range.
            rng (tuple,None)=None: optional explicit specification of target range. If None, use self.range
        Returns:
            True/False with respect to whether val is the right type and is within range. self._range is registered as side-effect
        '''
        if rng is None: rng = self.range
        if not isinstance( val, (int,float,str,Enum)) and len( val): # go per component
            isOk = []
            for i in range( len(self)):
                isOk.append( self.check_range( val[i], self.range[i]))
            return( all( isOk))
        else:
#             if hasattr( self, '_type') and isinstance( val, self.type): #??Maybe type checking should be deleted?
#                 print("TYPE confusion", val, type(val), self.type)
#                 try:
#                     val = self.type(val) # try to cast the new value
#                 except: # give up
#                     return( False)
            if isinstance( val, Enum):   return( isinstance( val, self.type))
            elif isinstance( val, str):  return( True) # no further requirements for str
            else:                        return(  rng is None or rng[0] <= val <= rng[1])
 
    def check_causality_variability_initial(self, initial: str, provideMsg: bool =True) -> bool:
        '''Check whether the combination of the defined self.causality + self.variability is allowed. See also FMI Specification page 51 of version 2.0.4
        We go through the case specification case by case A...E        
        Returns the initial value as Enum, if everyting consistent, otherwise None
        '''
        c = self.causality
        v = self.variability
        msg = ''
        if v==Variability.constant or c==Causality.parameter: # case (A) of page 51 of FMI 2.0.4
            if v==Variability.constant and c not in (Causality.output, Causaility.input):
                msg = "For variability 'constant', the causality shall be 'input' or 'output'. Found: " +c.name
            elif c==Causality.parameter and v not in (Variability.fixed, Variability.tunable):
                msg = "For causality 'parameter', the variability shall be 'fixed' or 'tunable'. Found: " +v.name
            else: # allowed combination A
                _initial = Initial.exact
                if not initial is None and initial != "exact" and provideMsg:
                    logger.error("When variability==constant or causality==parameter, initial must be set to 'exact'. This is fixed")
        elif v in (Variability.fixed, Variability.tunable): # case (B) of page 51 of FMI 2.0.4
            if c not in (Causality.calculatedParameter, Causality.local):
                msg = "For variability 'fixed' or 'tunable', the causality 'input', 'output' or 'independent' are not allowed. Found: " +c.name
            else: # allowed combination B
                if initial is None: _initial = Initial.calculated
                elif initial not in (Initial.calculated, Initial.approx): msg += "For variability 'fixed' or 'tunable' and causality 'calculated p.' or 'local', initial can only be 'calculated' or 'approx'. Found: " +str(initial)
        elif c == Causality.independent: # case (E) of page 51 of FMI 2.0.4
            if v != Variability.continuous:
                msg = "For causality 'continuous', the variability shall be 'independent'. Found: " +v.name
            else: # allowed combination E
                _initial = Initial.none
                if not initial is None and provideMsg:
                    logger.error("For variability 'continuous' and causality 'independent', initial must be left at 'none'. This is fixed")
        elif c == Causality.input: # case (D) of page 51 of FMI 2.0.4
            if v not in (Variability.discrete, Variability.continuous):
                msg = "For causality 'input', the variability shall be 'discrete' or 'continuous'. Found: " +v.name
            else: # allowed combination D
                _initial = Initial.none
                if not initial is None and provideMsg:
                    logger.error("For causality 'input', initial must be left at 'none'. This is fixed")
        elif c in (Causality.output, Causality.local): # case (C) of page 51 of FMI 2.0.4
            if v not in (Variability.discrete, Variability.continuous):
                msg = "For causality 'output' and 'local' the variability shall be 'discrete' or 'continuous'. Found: " +v.name
            else: # allowed combination C
                if initial is not None: # all values allowed
                    _initial = Initial[initial]
                else: 
                    _initial = Initial.exact # default value
        else:
            raise VariableInitError( "Causality/Variability/initial combination " +str(c)+" / " +str(v) +" / " +str(initial) + " not covered!")
        if len(msg):
            if provideMsg: logger.error( msg)
            return(None)
        else:
            return(_initial)
    
    def to_xml(self, **kwargs):
        """ Generate modelDescription XML code with respect to this variable, i.e. <ScalarVariable> element within <ModelVariables>.
        Note that ScalarVariable attributes should all be listed in __attrs dictionary
        Since we do not use the derived classes Real, ... we need to generate the detailed variable definitions here.
        The following attributes are so far not supported: declaredType, derivative, reinit 
        
        Args:
            **kwargs: additional keyword arguments (these override any properties):
            
            modelExchange: can be set to True, to signal a ModelExchange type model
            typ: the type can be explicitly provided (for derived variables), otherwise self._type is used
            name: a name can be explicitly provided (for components of derived variables)
            valueReference: a valueReference can be explicitly provided (for components of derived variables)
            initialVal: a initialVal can be explicitly provided (for components of derived variables)
            range: a range tuple can be explicitly provided (for components of derived variables)
            unit: a unit can be explicitly provided (for components of derived variables)
            displayUnit: a displayUnit can be explicitly provided (for components of derived variables)
            
        Returns:
            the etree element representing the sub-xml <ScalarVariable> tree for this variable
        """
        if kwargs.get('displayUnit', None) is None: kwargs['displayUnit'] = (self.unit, 1.0)
        typ = kwargs.get('typ', self._type)
        sv = super().to_xml()
        sv.attrib.update( { 'initial':self.initial.name, 'valueReference':str(self.valueReference if 'valueReference' not in kwargs else kwargs['valueReference'])})
        if 'name' in kwargs: sv.attrib.update( {'name':kwargs['name']})
        for a in ['name', 'valueReference', 'description', 'causality', 'variability', 'initial', 'valueReference']:
            if a not in sv.attrib:
                logger.warning("The attribute " +a +" is missing in variable " +self.name)
        if sv.attrib['initial'] == 'none': del sv.attrib['initial'] # the name 'none' is not officially defined
        if 'modelExchange' in kwargs and kwargs['modelExchange'] and self.variability==Variability.input:
            sv.attrib.update( { 'canHandleMultipleSetPerTimeInstant': self.canHandleMultipleSetPerTimeInstant})
        if len(self._annotations):
            an = ET.Element( 'annotations', self._annotations)
            sv.append( an)

        # detailed variable definition
        declaredType = {'int':'Integer', 'bool':'Boolean','float':'Real', 'str':'String', 'Enum':'Enumeration'}[typ.__qualname__] # translation of python to FMI primitives
        varInfo = ET.Element( declaredType)
        if ( self.initial in (Initial.exact, Initial.approx) or
             self.causality in (Causality.parameter, Causality.input) or
             self.variability == Variability.constant or
             ( self.causality in (Causality.output, Causality.local) and self.initial!=Initial.calculated)): # a start value is to be used
            varInfo.attrib.update( {'start':str( self.initialVal if 'initialVal' not in kwargs else kwargs['initialVal'])})
        if declaredType in ('Real', 'Integer', 'Enumeration'): # range to be specified
            xMin = self.range[0] if 'range' not in kwargs else kwargs['range'][0]
            if declaredType!='Real' or xMin>float('-inf'):
                varInfo.attrib.update( {'min': str(xMin)})
            else:
                varInfo.attrib.update( {'unbounded': 'true'})
            xMax = self.range[1] if 'range' not in kwargs else kwargs['range'][1]
            if declaredType!='Real' or xMax<float('inf'):
                varInfo.attrib.update( {'max': str(xMax)})
            else:
                varInfo.attrib.update( {'unbounded': 'true'})
        if declaredType == 'Real': # other attributes apply only to Real variables
            varInfo.attrib.update( {'unit': kwargs.get('unit', self.unit), 'displayUnit': kwargs.get('displayUnit', self.displayUnit)[0]})
            if varInfo.attrib['unit'] == varInfo.attrib['displayUnit']: del varInfo.attrib['displayUnit'] # unnecessary and causes problems
        sv.append( varInfo)
        return(sv)
                        
    def disect_unit(self, quantity: str|int|float|Enum):
        '''Disect the provided string in terms of magnitude and unit.

        Args:
            quantity (str): the quantity to disect. Should be provided as string, but also the trivial cases (int,float,Enum) are allowed. A free string should not be used and leads to a warning
        Returns:
            the magnitude in base units, the base unit and the unit as given together with the conversion factor (from displayUnit to baseUnit) '''
        if not isinstance( quantity, (int,float,str,Enum)):
            raise VariableInitError("Wrong value type " +str(type(quantity)) +" for scalar variable " +self.name +". Only primitive types are allowed.")
        if isinstance( quantity, str): # only string variable make sense to disect
            try:
                q = self.model.uReg( quantity) # parse the quantity-unit and return a Pint Quantity object
                if isinstance(q, (int,float)): return( q, '', None) # integer or float variable with no units provided
                elif isinstance(q, Quantity): # pint.Quantity object
                    displayUnit = str(q.units)
                    qB = q.to_base_units() # transform to base units ('SI' units). All internal calculations will be performed with these
                    if isinstance( self, Variable_NP): val = np.array( qB.magnitude, self._type).tolist()
                    else:                              val = qB.magnitude # Note: numeric types are not converted, e.g. int to float
                    uB = str(qB.units)
                    if displayUnit==uB:
                        return( val, uB, None)
                    else: # explicit displayUnits necessary
                        if val==0: # not suitable to identify the conversion factor
                            return( val, uB, (displayUnit, self.model.uReg.Quantity( 1.0, displayUnit).to_base_units().magnitude))
                        else:                    
                            return( val, uB, (displayUnit, val/q.magnitude))
                else:
                    raise VariableInitError("Unknown quantity " +quantity +" to disect")
            except: # no recognized units. Assume a free string. ??Maybe we should be more selective about the exact error type
                logger.warning("The string quantity " +quantity +" could not be disected. If it is a free string and explicit type 'typ=str' should be provided to avoid this warning") 
                return( quantity, '', None)
        else:
            return( quantity, 'dimensionless', None)
    
    def unit_convert(self, val:float|int|np.array|str, idx:int|None=None, toBase:bool|float=True):
        '''Convert the value 'val' between displayUnit and baseUnit

        Args:
            val: the value to be converted
            idx: Optional possibility to specify the component of a compound variable
            toBase=True: convert from displayUnit to baseUnit, False: from baseUnit to displayUnit, float: use the given factor
        
        self.displayUnit is either None (no conversions) or a tuple ( name of displayUnit, conversion factor toBase'''
        if isinstance( val, str): # single value provided as string with units
            val, bU, dU = self.disect_unit( val)
            if (idx is None and bU != self.unit) or (idx is not None and bU != self.unit[idx]):
                raise ValueError(f"The unit {bU} is not allowed for variable {self.name}")
            if toBase: return( val)            
        # val is numeric
        if isinstance( val, np.ndarray):
            print("UNIT_CONVERT", val, idx, toBase)
            if idx is None: # the whole variable
                value = [ self.unit_convert( val[i], idx=i, toBase=toBase) for i in range( len( self))]
                return( np.array( value, dtype=self._type))
            else: # an element of the array
                return( self.unit_convert( val[idx], idx=idx, toBase=toBase))
        else: # val is a single numeric value
            if idx is None and self.displayUnit is None: # no indexing and no conversion required
                return( val)
            elif idx is not None and self.displayUnit[idx] is None: # indexing and no conversion (for this index) required
                return( val)
            else:
                if idx is None: fac = self.displayUnit[1]
                else:           fac = self.displayUnit[idx][1]
                return( val*fac if toBase else val/fac)

    def getter(self): return( None) #Invalidate!. This function is not used within Variable!
    def setter(self, newVal): raise VariableUseError("The setter method shall not be used in connection with the Variable class")

                                                                                
class Variable_NP( Variable):
    ''' NumPy array variable as extension of Variable.
    The variable is internally kept as one object (with arrays of values, ranges, ...) and only when generating e.g. an FMU, the variable is split

        Args:
            model (obj): The model object where this variable relates to. Use model.add_variable( name, ...) to define variables
            name (str): Variable name, unique for whole FMU. The array components get names <name>.0,...
            description (str) = None: Optional description of variable. Array components get empty descriptions
            causality (str) = 'parameter': The causality setting as string. Same for whole array
            variability (str) = 'fixed': The variability setting as string. Same for whole array
            initial (str) = None: Definition how the variable is initialized. Provide this explicitly if the default value is not suitable. Same for the whole array
            canHandleMultipleSetPerTimeInstant (bool) = False: (only used for CoSimulation).
              Determines when in simulation it is allowed to set the variable, i.e. loops are not allowed when False. Same for whole array
            initialVal (tuple) = (): The initial value of the array components.
            
               This determines also the variable type of scalars in terms of Python primitive types.
            unit (Unit, tuple) = '': Optional unit of variable(s). If only one value is provided all components are assumed equal.
            rng (tuple) = (): Optional range of the array components. If only one tuple is provided, all components are assumed equal
            annotations (dict) = {}: Optional variable annotations provided as dict
            valueCheck (bool) = True: Optional possibility to bypass checking of new values with repect to type and range. Same for whole array
            typ (np.dtype) = None: Optional possibility to explicitly set the np.array type. Default: float64
            on_step (callable) = None: Optonal possibility to register a function of (currentTime, dT) to be run during Model.do_step,
               e.g. if the variable represents a speed, the object can be translated speed*dT, if |speed|>0
            on_set (callable) = None: Optional possibility to specify a function of (newVal) to be run when the variable is changed.
               This is useful for conditioning of input variables, so that calculations can be done once after a value is changed and do not need to be repeated on every simulation step.
               If given, the function shall apply to a value as expected by the variable (e.g. if there are components) and after unit conversion.
     '''
    def __init__(self, model:object, name: str, description:str=None, causality:str='parameter', variability:str='fixed', initial:str=None, canHandleMultipleSetPerTimeInstant: bool = False,
                 initialVal: tuple = (), unit: str|tuple = '', rng: tuple = tuple(),
                 annotations: dict = {}, valueCheck=VarCheck.all, typ=None, on_step:callable=None, on_set:callable=None):
        
        setattr(self, 'fullInit', False) # when calling super, the initialization is stopped where the array becomes relevant
        super().__init__( model=model, name=name, description=description, causality=causality, variability=variability, initial=initial,
                          canHandleMultipleSetPerTimeInstant=canHandleMultipleSetPerTimeInstant, annotations=annotations, valueCheck=valueCheck, on_step=on_step, on_set=on_set) # do basic initialization
        self._type = np.dtype('float64') if typ is None or not isinstance( typ, np.dtype) else typ
        self._len = len( initialVal)
        if not len(rng): rng = ((),)*self._len
        _initialVal, _unit, _displayUnit, _range = [], [], [], []
        for i in range( len(initialVal)):
            _i, _u, _d = self.disect_unit( initialVal[i]) # first disect the initial value
            _r = self.init_range( rng[i], _i, _u) if VarCheck.rangeCheck in self._valueCheck else None
            _initialVal.append( _i)
            _unit.append( _u)
            _displayUnit.append( _d)
            _range.append( _r)
        self._initialVal = tuple( _initialVal)
        self.unit = tuple( _unit)
        if all( d is None for d in _displayUnit):
            self.displayUnit = None # a quick way to state that no unit transformation is needed on the whole variable
        else:
            self.displayUnit = tuple( _displayUnit)
        self._range = tuple( _range)
        self._value = self.on_set( self._initialVal) if self.on_set is not None else np.array( self._initialVal, dtype=self._type) #pre-process and set the current value        
        self._idx = self.model.register_variable( self) # register in model and return index.
        #Note: Only the first valueReference is stored, representing the 0th element of the array
        self.valueReference = self._idx # always automatic reference number

    @property
    def type(self): return( type( np.array( 0, dtype=self._type).tolist())) # translate to the native python type
    
    @property
    def value(self): return( self._value)
    @value.setter
    def value(self, newVal):
        '''Set the variable (if allowed and if range check valid). This is designed for model-internal variable settings without units and not component-wise.
        If components of Variable_NP are set using .value[idx] = newVal, this invokes the @property above and checking and .on_set() are thus bypassed.
        Therefore, this setter function should be called with newVal=None after components are set.
        The 'Model._set' method is used for external (input) settings and deals with issues like components and units
        '''
        if newVal is None: newVal = self._value
        if not len(self) == len(newVal):
            raise VariableUseError("Erroneous dimension in new setting " +str(newVal) +" of variable " +self.name +". Expected dimension " +str( len( self)))
        if not all( [ self.check_range( v, rng=self._range[i]) for i,v in enumerate(newVal)]):
            raise VariableRangeError(f"Range violation in variable {self.name}, value {newVal}")
        self._value = np.array( newVal, dtype=self._type) if self.on_set is None else self.on_set( newVal)
#     def __getitem__(self, idx):
#         return( self._value[idx])
#     def __setitem__(self, idx, val):
#         if not super().check_range( val, rng=self._range[idx]):
#             raise VariableUseError(f"Variable {self.name} out of range. New value: {val}, range: {self._range[idx]}")
#         self._value = val
    
    def to_xml(self):
        scalarVars = []
        for i in range( len( self)):
            dU = None if self.displayUnit is None else self.displayUnit[i]
            scalarVars.append( super().to_xml( typ=float, name=self.name+'.'+str(i), valueReference=self.valueReference+i,
                                               initialVal=self.initialVal[i], range=self.range[i],
                                               unit=self.unit[i], displayUnit=(self.unit[i], 1.0) if dU is None else dU))
        return( scalarVars)
    
# Utility functions for handling special variable types
def spherical_to_cartesian( vec:np.array|tuple, asDeg:bool=False) -> np.array:
    '''Turn spherical vector 'vec' (defined according to ISO 80000-2 (r,polar,azimuth)) into cartesian coordinates'''
    if asDeg:
        theta = radians( vec[1])
        phi   = radians( vec[2])
    else:
        theta = vec[1]
        phi   = vec[2]
    sinTheta = sin( theta)
    cosTheta = cos( theta)
    sinPhi = sin( phi)
    cosPhi = cos( phi)
    r = vec[0]
    return( np.array( (r*sinTheta*cosPhi, r*sinTheta*sinPhi, r*cosTheta)))

def cartesian_to_spherical( vec:np.array, asDeg:bool=False) -> tuple:
    '''Turn the vector 'vec' given in cartesian coordinates into spherical coordinates
    (defined according to ISO 80000-2, (r, polar, azimuth))'''
    r = np.linalg.norm( vec)
    if vec[0]==vec[1]==0:
        if vec[2]==0: return( np.array( (0,0,0), dtype='float64'))
        else:         return( np.array( (r,0,0), dtype='float64'))
    elif asDeg:
        return( np.array( ( r, degrees( acos( vec[2]/r)), degrees(atan2( vec[1], vec[0]))), dtype='float64'))
    else:
        return( np.array( ( r, acos( vec[2]/r), atan2( vec[1], vec[0])), dtype='float64'))

def cylindrical_to_cartesian( vec:np.array|tuple, asDeg:bool=False) -> np.array:
    '''Turn cylinder coordinate vector 'vec' (defined according to ISO (r,phi,z)) into cartesian coordinates.
    The angle phi is measured with respect to x-axis, right hand'''
    phi = radians( vec[1]) if asDeg else vec[1]
    return( np.array((vec[0]* cos( phi), vec[0]* sin( phi), vec[2]), dtype='float64'))

def cartesian_to_cylindrical( vec:np.array, asDeg:bool=False) -> tuple:
    '''Turn the vector 'vec' given in cartesian coordinates into cylindrical coordinates
    (defined according to ISO, (r, phi, z), with phi right-handed wrt. x-axis)'''
    phi = atan2( vec[1], vec[0])
    if asDeg: phi = degrees( phi)
    return( np.array( ( sqrt( vec[0]*vec[0] + vec[1]*vec[1]), phi, vec[2])))


def quantity_direction( quantityDirection:tuple, asSpherical:bool=False, asDeg:bool=False) -> np.array:
    '''Turn a 4-tuple, consisting of quantity (float) and a direction 3-vector to a direction 3-vector,
    where the norm denotes the direction and the length denotes the quantity.
    The return vector is always a cartesian vector.
    
    Args:
        quantityDirection (tuple): a 4-tuple consisting of the desired length of the resulting vector (in standard units (m or m/s))
           and the direction 3-vector (in standard units)
        asSpherical (bool)=False: Optional possibility to provide the input direction vector in spherical coordinates
        asDeg (bool)=False: Optional possibility to provide the input angle (of spherical coordinates) in degrees. Only relevant if asSpherical=True
    '''
    if quantityDirection[0]<1e-15:
        return( np.array( (0,0,0), dtype='float64'))
    if asSpherical: direction = spherical_to_cartesian( quantityDirection[1:], asDeg) # turn to cartesian coordinates, if required
    else:           direction = np.array( quantityDirection[1:], dtype='float64')
    n = np.linalg.norm( direction) # normalize
    return( quantityDirection[0]/n*direction)
