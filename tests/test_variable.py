import sys, os
import math
import numpy as np
sys.path.insert(0, os.path.abspath('../'))
import logging
from component_model.logger import get_module_logger
logger = get_module_logger(__name__, level=logging.INFO)

from pythonfmu.enums import Fmi2Causality as Causality, Fmi2Initial as Initial, Fmi2Variability as Variability
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
#        print("ARR", arr1, arr2)
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

    def init_model_variables(self):
        '''Define model and a few variables for various tests'''
        mod = Model("MyModel")
        myInt = Variable( mod, "Test1", description="A integer variable", causality='parameter', variability='fixed', initialVal=99, rng=(0,100),  typ=int, annotations={}, valueCheck=VarCheck.all)
        myInt2 = Variable( mod, "Test2", description="A integer variable without range checking", causality='input', variability='continuous', initialVal=99, rng=(0,100), typ=int, annotations={}, valueCheck=VarCheck.none)
        myFloat = mod.add_variable( "Test3", description="A float variable", causality='input', variability='continuous', initialVal='99.0%', rng=(0.0,None),  annotations={})
        myEnum = Variable( mod, "Test4", description="An enumeration variable", causality='output', variability='discrete', initialVal=Causality.parameter,  annotations={})
        myStr = Variable( mod, "Test5", description="A string variable", typ=str, causality='parameter', variability='fixed', initialVal="Hello World!",  annotations={})
        myNP = Variable_NP( mod, "Test6", description="A NP variable", causality='parameter', variability='fixed', initialVal= ('1.0m','2deg','3rad'),  rng=( (0,'3m'), (0,float('inf')), (float('-inf'),'5rad')))
        return( mod,myInt,myInt2,myFloat,myEnum,myStr,myNP)
        
    def test_init(self):
        mod,myInt,myInt2,myFloat,myEnum,myStr,myNP = self.init_model_variables()
        # test _get_auto_extreme()
        self.assertEqual( Variable._get_auto_extreme( 1.0), ( float('-inf'), float('inf')))
        with self.assertRaises( VariableInitError) as err:
            print( Variable._get_auto_extreme( 1))
        self.assertEqual( str( err.exception), "Unlimited integer variables do not make sense in Python. Please provide explicit limits for variable 1 or set the type to float.")
        self.assertEqual( myInt.range, (0,100))
        self.assertEqual( myInt.name, "Test1")
        self.assertEqual( myInt.value, 99)
        myInt.value = 50
        self.assertEqual( myInt.value, 50)
        with self.assertRaises( VariableRangeError) as err:
            myInt.value = 101
        self.assertTrue( str(err.exception).startswith('The value 101 is not accepted within variable Test1'))
        myInt2.value = 101 # does not cause an error message

        with self.assertRaises( ModelInitError) as err:
            myInt3 = Variable( mod, "Test1", description = "An integer variable with a non-unique name", causality='input', variability='continuous', typ=int, canHandleMultipleSetPerTimeInstant = False,
                               initialVal='99.9%', rng=(0,'100%'),  annotations={})
        self.assertTrue( str(err.exception).startswith("Variable name Test1 is not unique"))

        with self.assertRaises( VariableInitError) as err:
            myInt = Variable( mod, "Test8", description="A second integer variable with erroneous range", causality='parameter', variability='fixed', initialVal='99', rng=(),  annotations={}, typ=int)
        self.assertEqual( str( err.exception), "Unlimited integer variables do not make sense in Python. Please provide explicit limits for variable 99 or set the type to float.")

        # one example using add_variable on the model
        self.assertEqual( myFloat.range[1], 0.99)
        
        self.assertEqual( myEnum.range, (0,5))
        self.assertTrue( myEnum.check_range( Causality.parameter))
        
        self.assertEqual( len(myStr.range), 0)
        

    def test_variable_np(self):
        mod = Model("MyModel")
        myNP = Variable_NP( mod, "Test6", description="A NP variable", causality='parameter', variability='fixed', initialVal= ('1.0m','2deg','3rad'),  rng=( (0,'3m'), (0,float('inf')), (float('-inf'),'5rad')))
        self.assertEqual( myNP.initialVal, (1,math.radians(2),3))
        self.assertEqual( myNP.name, "Test6")
        self.assertEqual( myNP.range[1], (0,float('inf')))
        self.assertEqual( str(myNP.displayUnit[1][0]), 'degree')
        self.np_arrays_equal( myNP.value, np.array( (1.0,math.radians(2.0),3.0), dtype='float64')) 
        myNP.value = np.array( (1.5, 2.5, 3.5), dtype='float64')
        self.np_arrays_equal( myNP.value, np.array( (1.5,2.5,3.5), dtype='float64'))
        self.assertEqual( np.linalg.norm( myNP.value), math.sqrt( 1.5**2 + 2.5**2 + 3.5**2)) #np calculations are done on value
        myNP.value[0] = 15 # does not cause a error
        with self.assertRaises( VariableRangeError) as err:
            myNP.value = None # check the new value and run .on_set if defined, causing an error
        self.assertTrue( str( err.exception).startswith("Range violation in variable Test6"))
        myNP2 = Variable_NP( mod, "Test9", description="A NP variable with units included in initial values and partially fixed range", causality='input', variability='continuous',
                             initialVal= ('1m','2deg','3 deg'),  rng=( (0,'3m'), None, None))
        self.assertEqual( tuple(myNP2.initialVal), (1,math.radians(2),math.radians(3)))
        self.assertEqual( str(myNP2.unit[0]), 'meter')
        self.assertEqual( tuple(myNP2.range[0]), (0,3))
        self.assertEqual( tuple(myNP2.range[1]), (math.radians(2),math.radians(2)))

        myFloat2 = Variable( mod, "Test7", description="A float variable with units included in initial value",
                             causality='parameter', variability='fixed', initialVal='99%', rng=(0.0,None),  annotations={})
        self.assertEqual( myFloat2.initialVal, 0.99)
        self.assertEqual( str(myFloat2.displayUnit[0]), 'percent')
        myFloat3 = Variable( mod, "Test8", description="A float variable with delta range", causality='parameter', variability='fixed', initialVal='99%', rng=None,  annotations={})
        self.assertEqual( myFloat3.range, (0.99, 0.99))
        
        boom = Variable_NP( mod, "test-boom", "The dimension and direction of the boom from anchor point to anchor point in m and spherical angles", causality='input', variability='continuous',
                                      initialVal = ('5.0 m', '-90 deg', '0deg'), rng = (None, ('-180deg','180deg'), None))
        print("BOOM", boom.initialVal, np.linalg.norm( boom.initialVal))

    def test_var_ref(self):
        mod,myInt,myInt2,myFloat,myEnum,myStr,myNP = self.init_model_variables()
        self.assertEqual( mod.vars[1].name, "Test2")
        self.assertIsNone( mod.vars[6]) #a sub-element
        var,sub = mod.ref_to_var( 6)
        self.assertEqual(var.name, "Test6")
        self.assertEqual( sub,1)
        self.assertEqual( mod.variable_by_name("Test2").name, "Test2")
        self.assertEqual( mod.variable_by_name("Test2").valueReference, 1)
        
    def test_vars_iter(self):
        mod,myInt,myInt2,myFloat,myEnum,myStr,myNP = self.init_model_variables()
        self.assertEqual( len(list(mod.vars_iter(float))), 2)
        self.assertEqual( list( mod.vars_iter(float))[0].name, 'Test3')
        self.assertEqual( list( mod.vars_iter(float))[1].name, "Test6")
        self.assertEqual( list( mod.vars_iter( key=Variability.discrete))[0].name, 'Test4')
        self.assertEqual( list( mod.vars_iter( key=Causality.input))[1].name, 'Test3') 
        self.assertEqual( list( mod.vars_iter( key=lambda x: x.causality==Causality.input or x.causality==Causality.output))[2].name, 'Test4')
    
    def test_get(self):
        mod,myInt,myInt2,myFloat,myEnum,myStr,myNP = self.init_model_variables()
        self.assertEqual( mod._get( [0,1], int), [99,99])  
        self.assertEqual( mod.get_integer( [0,1]), [99,99])     
        self.assertEqual( mod.get_integer( [0,1]), [99,99])     
        with self.assertRaises( TypeError) as err:
            vals = mod.get_real( [0,1])
        self.assertTrue( str( err.exception).startswith("Variable with valueReference="))
        self.assertEqual( mod.get_real( [2]), [99])
        print( mod.get_real([5]))
        self.assertEqual( mod.get_real([5]), [1.0])
        self.assertEqual( mod.get_real([5,6]), [1,2])
#         with self.assertRaises( KeyError) as err:
#             vals = mod.get_real([5,8])
#         self.assertTrue( str( err.exception).startswith("Variable with valueReference=8 does not exist in model My"))

    def test_set(self):
        mod,myInt,myInt2,myFloat,myEnum,myStr,myNP = self.init_model_variables()
        mod.set_integer( [0,1], [50, 51])
        self.assertEqual( mod.vars[0].value, 50)
        self.assertEqual( mod.vars[1].value, 51)
        with self.assertRaises( TypeError) as err:
            mod.set_integer( [6,7], [2.0, '30 deg'])
        self.assertTrue( str(err.exception).startswith("Variable with valueReference=6 is not of type int"))
        mod.set_real( [6,7], [2.0, '30 deg'])

        
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
    suite.addTest( Test_variable("test_var_ref"))
    suite.addTest( Test_variable("test_vars_iter"))
    suite.addTest( Test_variable("test_get"))
    suite.addTest( Test_variable("test_set"))
    
    test_result = unittest.TextTestRunner(verbosity=1).run(suite)
    if test_result.wasSuccessful():
        pass # possibility to perform cleanup when everything was successful
        #logger.info("Ran " +str( test_result.testsRun) +" tests. Success")
    else:
        print( test_result)

