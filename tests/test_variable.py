import sys, os
import math
import numpy as np
sys.path.insert(0, os.path.abspath('../'))
import logging
from component_model.logger import get_module_logger
logger = get_module_logger(__name__, level=logging.INFO)

from pythonfmu.enums import Fmi2Causality as Causality
from component_model.model import Model, ModelInitError
from component_model.variable import Variable, Variable_NP, VariableInitError, VariableRangeError, VariableUseError, spherical_to_cartesian, cartesian_to_spherical, VarCheck

import unittest


class Test_variable( unittest.TestCase):
    """ Test suite for the variable module """                         

    def test_varCheck(self):
        ck = VarCheck.unitNone | VarCheck.rangeNone
        self.assertTrue( VarCheck.rangeNone in ck)
        self.assertTrue( VarCheck.unitNone in ck)
        ck = VarCheck.unitAll | VarCheck.rangeCheck
        self.assertTrue( VarCheck.rangeCheck in ck)
        self.assertTrue( VarCheck.unitAll in ck)
        self.assertTrue( ck & VarCheck.units | VarCheck.unitAll) # filter the combined flag on unit
        self.assertTrue( ck & VarCheck.ranges | VarCheck.rangeCheck) # filter the combined flag on range    

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
        mod = Model("MyModel")
        # test _get_auto_extreme()
        self.assertEqual( Variable._get_auto_extreme( 1.0), ( float('-inf'), float('inf')))
        with self.assertRaises( VariableInitError) as err:
            print( Variable._get_auto_extreme( 1))
        self.assertEqual( str( err.exception), "Unlimited integer variables do not make sense in Python. Please provide explicit limits for variable 1 or set the type to float.")
        myInt = Variable( mod, "Test1", description="A integer variable", causality='parameter', variability='fixed', initialVal=99, rng=(0,100),  annotations={}, valueCheck=VarCheck.all)
        self.assertEqual( myInt.range, (0,100))
        self.assertEqual( myInt.name, "Test1")
        self.assertEqual( myInt.getter(), 99)
        self.assertEqual( myInt.value, 99) #two alternative ways of accessing _value
        myInt.value = 50
        self.assertEqual( myInt.value, 50)
        with self.assertRaises( VariableRangeError) as err:
            myInt.value = 101
        self.assertTrue( str(err.exception).startswith('The value 101 is not accepted within variable Test1'))
        myInt2 = Variable( mod, "Test2", description="A integer variable without range checking", causality='parameter', variability='fixed', initialVal=99, rng=(0,100), annotations={}, valueCheck=VarCheck.none)
        myInt2.value = 101 # does not caus an error message

        with self.assertRaises( ModelInitError) as err:
            myFloat = Variable( mod, "Test1", description = "A float variable with a non-unique name", causality='input', variability='continuous', valueReference = None, canHandleMultipleSetPerTimeInstant = False,
                               initialVal='99.9%', rng=(0,'100%'),  annotations={})
        self.assertEqual( str(err.exception), "Variable name Test1 is not unique")

        with self.assertRaises( VariableInitError) as err:
            myInt = Variable( mod, "Test2", description="A second integer variable with erroneous range", causality='parameter', variability='fixed', initialVal='99', rng=(),  annotations={})
        self.assertEqual( str( err.exception), "Unlimited integer variables do not make sense in Python. Please provide explicit limits for variable 99 or set the type to float.")

        # one example using add_variable on the model
        myFloat = mod.add_variable( "Test3", description="A float variable", causality='parameter', variability='fixed', initialVal='99.0%', rng=(0.0,None),  annotations={})
        self.assertEqual( myFloat.range[1], 0.99)
        
        myEnum = Variable( mod, "Test4", description="An enumeration variable", causality='parameter', variability='fixed', initialVal=Causality.parameter,  annotations={})
        print("RANGE", myEnum.range)
        self.assertTrue( myEnum.check_range( Causality.parameter))
        
        myStr = Variable( mod, "Test5", description="A string variable", typ=str, causality='parameter', variability='fixed', initialVal="Hello World!",  annotations={})
        self.assertEqual( len(myStr.range), 0)

    def test_variable_np(self):
        mod = Model("MyModel")
        myNP = Variable_NP( mod, "Test6", description="A NP variable", causality='parameter', variability='fixed', initialVal= ('1.0m','2deg','3rad'),  rng=( (0,'3m'), (0,float('inf')), (float('-inf'),'5rad')))
        self.assertEqual( mod.variables[0].name, "Test6")
        self.assertEqual( myNP.initialVal, (1,math.radians(2),3))
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
        self.assertTrue( str( err.exception).startswith("The value 15 is not accepted within variable Test6"))
        myNP2 = Variable_NP( mod, "Test9", description="A NP variable with units included in initial values and partially fixed range", causality='input', variability='continuous',
                             initialVal= ('1m','2deg','3 deg'),  rng=( (0,'3m'), None, None))
        self.assertEqual( tuple(myNP2.initialVal), (1,math.radians(2),math.radians(3)))
        self.assertEqual( str(myNP2.unit[0]), 'meter')
        self.assertEqual( tuple(myNP2.range[0]), (0,3))
        self.assertEqual( tuple(myNP2.range[1]), (math.radians(2),math.radians(2)))

        myFloat2 = Variable( mod, "Test7", description="A float variable with units included in initial value",
                             causality='parameter', variability='fixed', initialVal='99%', rng=(0.0,None),  annotations={})
        self.assertEqual( myFloat2.initialVal, 0.99)
        self.assertEqual( str(myFloat2._displayUnit), 'percent')
        myFloat3 = Variable( mod, "Test8", description="A float variable with delta range", causality='parameter', variability='fixed', initialVal='99%', rng=None,  annotations={})
        self.assertEqual( myFloat3.range, (0.99, 0.99))
        
        boom = Variable_NP( mod, "test-boom", "The dimension and direction of the boom from anchor point to anchor point in m and spherical angles", causality='input', variability='continuous',
                                      initialVal = ('5.0 m', '-90 deg', '0deg'), rng = (None, ('-180deg','180deg'), None))
        print("BOOM", boom.initialVal, np.linalg.norm( boom.initialVal))

    def test_xml(self):
        mod = Model("MyModel")
        myInt = Variable( mod, "Test1", description="A integer variable", causality='parameter', variability='fixed', initialVal='99%', rng=(0,'100%'),  annotations={}, valueCheck=VarCheck.all)
        myInt.to_xml()
        myNP2 = Variable_NP( mod, "Test9", description="A NP variable with units included in initial values and partially fixed range", causality='input', variability='continuous',
                             initialVal= ('1m','2deg','3 deg'),  rng=( (0,'3m'), None, None))
        myNP2.to_xml()
        
if __name__ == '__main__':
#    suite = unittest.TestLoader().loadTestsFromTestCase(Test_variable) # use that to do all tests
    suite = unittest.TestSuite() # use this to load only single tests (together with next lines)
    # single tests:
    suite.addTest( Test_variable("test_varCheck"))
    suite.addTest( Test_variable("test_init")) 
    suite.addTest( Test_variable("test_variable_np")) 
    suite.addTest( Test_variable("test_spherical_cartesian")) 
    suite.addTest( Test_variable("test_xml"))
    test_result = unittest.TextTestRunner(verbosity=1).run(suite)
    if test_result.wasSuccessful():
        pass # possibility to perform cleanup when everything was successful
        #logger.info("Ran " +str( test_result.testsRun) +" tests. Success")
    else:
        print( test_result)

