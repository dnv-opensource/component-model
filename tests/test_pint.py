'''Test the pint package and identify the functions we need for this package'''
import sys, os
import numpy as np
sys.path.insert(0, os.path.abspath('../'))
import logging
from crane_fmu.logger import get_module_logger
logger = get_module_logger(__name__, level=logging.INFO)

from pint import UnitRegistry, OffsetUnitCalculusError
uReg = UnitRegistry( system='SI', autoconvert_offset_to_baseunit=True)#, auto_reduce_dimensions=True)
 
import unittest

class Test_pint( unittest.TestCase):
    """ Test suite for the utils module """                         

    def test_needed_functions(self):
        uReg = UnitRegistry( system='SI', autoconvert_offset_to_baseunit=True)#, auto_reduce_dimensions=True)
        print("AVAILABLE UNITS", dir(uReg.sys.SI))
        print("degrees_Celsius defined?", 'degree_Celsius' in uReg, " but not degrees_Celsius: ", 'degrees_Celsius' in uReg)
        print("Unit System", uReg.default_system)
        print("Implicit and explicit def", 0.2*uReg.kg, uReg.Quantity(0.2, uReg.kg))
        print("String parsing:", uReg("0.2kg"), uReg(" 0.2 kg"), uReg("kg"), uReg("1.3e5"))	
        print("Base Units", uReg.Quantity( 1.0, 'ft').to_base_units(), uReg("0.2 kg").to_base_units())
        print("Disect", uReg.Quantity(1.0, 'ft').magnitude, uReg.Quantity(1.0, 'ft').units, uReg.Quantity(1.0, 'ft').dimensionality)
        print("Temperature", uReg.Quantity( 38.0, uReg.degC), uReg("38.0*degK"), uReg.Quantity( 38.0, "degF"), uReg.Quantity( 38.0, "degF").to_base_units()) #string recognition works only for Kelvin, because the others have an offset
        q = uReg("38 degC")
        print("Temperature from string", q, q.magnitude, q.units, q.dimensionality, q.to_base_units())
        u0 = str( q.units)
        qB = q.to_base_units()
        val = qB.magnitude
        uB = str(qB.units)
        qInv = uReg.Quantity( val, uB)
        print("QINV", type( qInv))
        self.assertEqual( q, qInv.to( 'degC'))
        q = uReg("36 deg") # angle
        self.assertEqual( str(q.to_base_units().units), 'radian')
        self.assertEqual( str(q.units), 'degree')
        self.assertEqual( str(uReg("1.0%").to_base_units().units), 'dimensionless')
        q = uReg("3.6 N")
        print( "dimensionality of N: ", ''.join( x +':'+ str(q.dimensionality[x]) +', ' for x in q.dimensionality))
        self.assertEqual( q.dimensionality['[time]'], -2)
        self.assertEqual( q.units, 'newton')
        print( q.to_base_units().units)
        print( q.to_reduced_units())
        self.assertTrue( q.check( { '[mass]':1, '[length]':1, '[time]':-2}))
        self.assertTrue( q.check('[force]')) # it is also a force (derived dimension)
        q = uReg("2 m^2")
        #print("COMPATIBLE UNITS: ", uReg.get_compatible_units(q ))
        self.assertTrue( q.is_compatible_with( 'square_mile')) # check compatibility with a given unit
        self.assertTrue( q.check( '[area]')) # check compatibility with derived dimensions
        q = uReg("1 mol")
        print("SUBSTANCE:", q, q.to_base_units(), q.dimensionality)
        q = uReg("100 cd")
        print("LUMINOSITY:", q, q.to_base_units(), q.dimensionality, uReg.get_base_units(list(uReg.get_compatible_units(q.dimensionality))[0]))
        q = uReg("9 degrees/s")
        print("RAD/s:", q.check('rad/s'), q.check('s/rad'), q, q.to_base_units(), q.dimensionality)
        q = uReg("9 s/deg")
        print("RAD/s:", q.check('rad/s'), q.check({'s':1, 'rad':-1}), q, q.to_base_units(), q.dimensionality)
        
        

    def test_split(self):
        self.assertEqual( Unit.split( 'kg'), (1.0,'kg'))
        self.assertEqual( Unit.split( '0.2kg'), (0.2,'kg'))
        self.assertEqual( Unit.split( '0.2 kg'), (0.2, 'kg'))
        self.assertEqual( Unit.split( '0.2E-4 m'), (2E-5, 'm'))
        self.assertEqual( Unit.split( 0.2E-4), (2E-5, ''))
        self.assertEqual( tuple( [ Unit.split( u) for u in ('3m','45 deg', 0)]), ((3,'m'),(45,'deg'),(0,'')))

    def test_quantity(self):
        self.assertEqual( Unit.quantity("30%"), (0.3, '%', 'percent'))
        self.assertEqual( Unit.quantity("m"), (1.0, 'm', 'length'))        
        self.assertEqual( Unit.quantity("-10 deg"), (-0.17453292519943295, 'deg', 'angle'))
        with self.assertRaises( UnitError): Unit.quantity("0.2um") # undefined unit
        self.assertEqual( Unit.quantity("0.2kg", 'mass'), (0.2, 'kg', 'mass'))
        with self.assertRaises( UnitError): Unit.quantity("0.2kg", 'length') # unit not as expected

    def test_convert(self):
        self.assertAlmostEqual( Unit.convert( 1.0, 'nm'), 0.0005399568034557236)
        self.assertAlmostEqual( Unit.convert( "3kg", 'lb'), 6.613867865546327)
    
    def test_get_standard_unit(self):
        self.assertEqual( Unit.get_standard_unit( 'mass'), 'kg')

if __name__ == '__main__':
#    suite = unittest.TestLoader().loadTestsFromTestCase(Test_utils) # use that to do all tests
    suite = unittest.TestSuite() # use this to load only single tests (together with next lines)
    # single tests:
    suite.addTest( Test_pint( "test_needed_functions"))
#     suite.addTest( Test_pint("test_split")) 
#     suite.addTest( Test_pint("test_quantity"))
#     suite.addTest( Test_pint("test_convert"))
#     suite.addTest( Test_pint("test_get_standard_unit"))
    test_result = unittest.TextTestRunner(verbosity=1).run(suite)
    if test_result.wasSuccessful():
        pass # possibility to perform cleanup when everything was successful
        #logger.info("Ran " +str( test_result.testsRun) +" tests. Success")
    else:
        print( test_result)

