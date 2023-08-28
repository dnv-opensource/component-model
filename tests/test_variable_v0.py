import sys, os
import math
import numpy as np
sys.path.insert(0, os.path.abspath('../'))
import logging
from component_model.logger import get_module_logger
logger = get_module_logger(__name__, level=logging.INFO)

from pint import UnitRegistry, Quantity, UndefinedUnitError # management of units
from pythonfmu.enums import Fmi2Causality as Causality, Fmi2Initial as Initial, Fmi2Variability as Variability
from component_model.variable import Variable, Variable_NP, VariableInitError, VariableRangeError, VariableUseError, spherical_to_cartesian, cartesian_to_spherical
from component_model.model import Model, ModelInitError

import unittest


class Test_variable( unittest.TestCase):
    """ Test suite for the variable module """                         


    def np_arrays_equal(self, arr1, arr2, dtype='float64'):
        self.assertTrue( len(arr1) == len(arr2), "Length not equal!")
        if isinstance( arr2, tuple): arr2 = np.array( arr2, dtype=dtype)
        self.assertTrue( isinstance( arr1, np.ndarray) and isinstance( arr2, np.ndarray), "At least one of the parameters is not an ndarray!")
        self.assertTrue( arr1.dtype==arr2.dtype,  "The two arrays do not have the same dtype: " +str(arr1.dtype) +"!=" +str(arr2.dtype))
        for i in range( len( arr1)):
            self.assertAlmostEqual( arr1[i], arr2[i], msg="Component " +str(i) +" is not equal for the two arrays. Found: " +str(arr1[i]) +"!=" +str(arr2[i])+"!")

    def test_spherical_cartesian(self):
        for vec in [ (0,0,0), (0,0,1), (0,1,0), (0,1,1), (1,0,0), (1,0,1), (1,1,0), (1,1,1)]:
            sVec = cartesian_to_spherical( vec)
            _vec = spherical_to_cartesian( sVec)
            self.np_arrays_equal( np.array(vec, dtype='float64'), _vec)

    def test_init(self):
        uReg = UnitRegistry( system='SI', autoconvert_offset_to_baseunit=True)

        # test _get_auto_extreme()
        self.assertEqual( Variable._get_auto_extreme( 1.0), ( float('-inf'), float('inf')))
        with self.assertRaises( VariableInitError) as err:
            print( Variable._get_auto_extreme( 1))
        self.assertEqual( str( err.exception), "Unlimited integer variables do not make sense in Python. Please provide explicit limits for variable 1 or set the type to float.")
        myInt = Variable( "Test1", description="A integer variable", causality='parameter', variability='fixed', initialVal=99, rng=(0,100),  annotations={}, valueCheck=True)
        self.assertEqual( myInt.range, (0,100))
        self.assertEqual( myInt.name, "Test1")
        self.assertEqual( myInt.value, 99)
        myInt.value = 50
        self.assertEqual( myInt.value, 50)
        with self.assertRaises( VariableRangeError) as err:
            myInt.value = 101
        self.assertEqual( str(err.exception), 'The value 101 is not accepted within variable Test1')

        with self.assertRaises( VariableInitError) as err:
            myInt = Variable( "Test2", description="A second integer variable with erroneous range", causality='parameter', variability='fixed', initialVal='99', rng=(),  annotations={}, uReg=uReg)
        self.assertEqual( str( err.exception), "Unlimited integer variables do not make sense in Python. Please provide explicit limits for variable 99 or set the type to float.")

        myFloat = Variable( "Test3", description="A float variable", causality='parameter', variability='fixed', initialVal='99.0%', rng=(0.0,None),  annotations={}, uReg=uReg)
        self.assertEqual( myFloat.range[1], 0.99)
        
        myEnum = Variable( "Test4", description="An enumeration variable", causality='parameter', variability='fixed', initialVal=Causality.parameter,  annotations={})
        self.assertTrue( myEnum.check_range( Causality.parameter))
        
        myStr = Variable( "Test5", description="A string variable", typ=str, causality='parameter', variability='fixed', initialVal="Hello World!",  annotations={}, uReg=uReg)
        self.assertEqual( len(myStr.range), 0)
        
        myFloat = Variable( "Test6", description="A float variable with explicit objects during instantiation",
                            causality = Causality['parameter'], variability = Variability['fixed'], initial = Initial['exact'],
                            initialVal = Quantity( 99.0, '%'), rng=( Quantity(0.0,'%'), Quantity(100.0,'%')),  annotations={}, uReg=uReg)
        self.assertEqual( myFloat.range[1], 1.0)

    def test_variable_np(self):
        uReg = UnitRegistry( system='SI', autoconvert_offset_to_baseunit=True)
        with self.assertRaises( VariableInitError) as err:
            myNP = Variable_NP( "Test6", description="A NP variable", causality='parameter', variability='fixed', initialVal= ('1.0m','2deg','3rad'),  annotations={}, rng=( (0,3), (0,float('inf')), (float('-inf'),5)), uReg=uReg)
            self.assertTrue( str( err.exception).startswith("No units provided for range (0, 3) of variable Test6"))
        myNP = Variable_NP( "Test6", description="A NP variable", causality='parameter', variability='fixed', initialVal= ('1.0m','2deg','3rad'),  annotations={}, rng=( (0,'3m'), (0,float('inf')), (float('-inf'),'5rad')), uReg=uReg)
        #print( ''.join( e.name+', ' for e in Variable.variableList))
        self.np_arrays_equal( myNP.initialVal, np.array( (1,math.radians(2),3), dtype='float64'))
        #print("SUB", ''.join( str(s)+", " for s in myNP._sub))
        self.assertEqual( myNP.name, "Test6")
        self.assertEqual( myNP.range[1], (0,float('inf')))
        self.assertEqual( str(myNP._displayUnit[1]), 'degree')
        self.np_arrays_equal( myNP.value, np.array([1,math.radians(2),3.0]))
        print( myNP.value)
        myNP.value = (1.5, 2.5, 3.5)
        self.np_arrays_equal( myNP.value, np.array( (1.5,2.5,3.5), dtype='float64'))
        with self.assertRaises( VariableRangeError) as err:
            myNP.value = (15, 2.5, 3.5)
        self.assertEqual( str( err.exception), "The value (15, 2.5, 3.5) is not accepted within variable Test6")
        myNP2 = Variable_NP( "Test9", description="A NP variable with units included in initial values and partially fixed range", causality='input', variability='continuous',
                             initialVal= ('1m','2deg','3 deg'),  rng=( (0,'3m'), None, None), uReg=uReg)
        self.assertEqual( tuple(myNP2.initialVal), (1,math.radians(2),math.radians(3)))
        self.assertEqual( str(myNP2.unit[0]), 'meter')
        self.assertEqual( tuple(myNP2.range[0]), (0,3))
        self.assertEqual( tuple(myNP2.range[1]), (math.radians(2),math.radians(2)))

        myFloat2 = Variable( "Test7", description="A float variable with units included in initial value",
                             causality='parameter', variability='fixed', initialVal='99%', rng=(0.0,None),  annotations={}, uReg=uReg)
        self.assertEqual( myFloat2.initialVal, 0.99)
        self.assertEqual( str(myFloat2._displayUnit), 'percent')
        myFloat3 = Variable( "Test8", description="A float variable with delta range", causality='parameter', variability='fixed', initialVal='99%', rng=None,  annotations={}, uReg=uReg)
        self.assertEqual( myFloat3.range, (0.99, 0.99))
        
        boom = Variable_NP( "test-boom", "The dimension and direction of the boom from anchor point to anchor point in m and spherical angles", causality='input', variability='continuous',
                                      initialVal = ('5.0 m', '-90 deg', '0deg'), rng = (None, ('-180deg','180deg'), None), uReg=uReg)
        print("BOOM", boom.initialVal, np.linalg.norm( boom.initialVal))

    def test_variable_model(self):
        '''Test variables in the context of a model'''
        mod = Model("TestModel", description="A model for testing variables",)
        myInt = Variable( "Test1", description="A integer variable", causality='parameter', variability='fixed', initialVal=99, rng=(0,100),  annotations={}, valueCheck=True)
        mod.add_variable( myInt)
        self.assertEqual( mod.variables[0].name, "Test1")
        self.assertEqual( mod._refCount, 1)
        with self.assertRaises( VariableInitError) as err:
            myInt2 = Variable( "Test1", description="A second integer variable with erroneous range", causality='parameter', variability='fixed', initialVal='99', rng=(),  annotations={})
            self.assertTrue( str(err.exception).startswith( "Unlimited integer variables do not make sense in Python"))
        myInt2 = Variable( "Test1", description="A second integer variable with erroneous range", causality='parameter', variability='fixed', initialVal='99', rng=(0,100),  annotations={})
        with self.assertRaises( ModelInitError) as err:
            mod.add_variable( myInt2)
            self.assertEqual( str(err.exception), "Variable Test1 is already in use")

        Variable_NP( "Test6", description="A NP variable", causality='parameter', variability='fixed', initialVal= ('1.0m','2deg','3rad'),  annotations={}, rng=( (0,'3m'), (0,float('inf')), (float('-inf'),'5rad'))).add_to( mod)
        self.assertEqual( mod.variables[1].name, "Test6")

    def test_xml(self):
        '''Test the to_xml function, which generates the 'ScalarVariable' element for a given variable'''
        mod = Model("TestModel", description="A model for testing variables",)
        Variable( "Test1", description="A integer variable", causality='parameter', variability='fixed', initialVal=99, rng=(0,100),  annotations={}, valueCheck=True).add_to( mod)
        Variable_NP( "Test6", description="A NP variable", causality='parameter', variability='fixed', initialVal= ('1.0m','2deg','3rad'),  annotations={}, rng=( (0,'3m'), (0,float('inf')), (float('-inf'),'5rad'))).add_to( mod)
        mod.variables[1].to_xml()
         

if __name__ == '__main__':
#    suite = unittest.TestLoader().loadTestsFromTestCase(Test_variable) # use that to do all tests
    suite = unittest.TestSuite() # use this to load only single tests (together with next lines)
    # single tests:
#    suite.addTest( Test_variable("test_init")) 
#    suite.addTest( Test_variable("test_variable_model"))
    suite.addTest( Test_variable("test_variable_np")) 
#    suite.addTest( Test_variable("test_spherical_cartesian")) 
#    suite.addTest( Test_variable("test_xml"))
    test_result = unittest.TextTestRunner(verbosity=1).run(suite)
    if test_result.wasSuccessful():
        pass # possibility to perform cleanup when everything was successful
        #logger.info("Ran " +str( test_result.testsRun) +" tests. Success")
    else:
        print( test_result)

