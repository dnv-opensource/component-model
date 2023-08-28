import sys, os
import numpy as np
sys.path.insert(0, os.path.abspath('../'))
import logging
from crane_fmu.logger import get_module_logger
logger = get_module_logger(__name__, level=logging.INFO)

from crane_fmu.model import Model
from crane_fmu.variable import Variable

import unittest
import xml.etree.ElementTree as ET

class Test_Model( unittest.TestCase):
    """ Test suite for the Model module """                         

    def test_init(self):
        mod = Model( "MyModel", )
        print(mod.description)
        mod.add_variable( "Test1", description="A length variable", causality='parameter', variability='fixed', initialVal='9 m', rng=('1m','20m'),  annotations={})
        mod.add_variable( "Test2", description="A force variable", causality='parameter', variability='fixed', initialVal='7 N', rng=('0','100N'),  annotations={})
        mod.add_variable( "Test3", description="An angular velocity variable", causality='parameter', variability='fixed', initialVal='30 m/deg**2/s', rng=('0','1000000 m/deg**2/h'),  annotations={})
        #print(mod.variables)
#        print( ElementTree.tostring( mod.make_unit_definitions()))
        print( ET.dump(mod.make_unit_definitions()))
#         pedestal = Boom( name         ='pedestal',
#                          description  = "The crane base, on one side fixed to the vessel and on the other side the first crane boom is fixed to it. The mass should include all additional items fixed to it, like the operator's cab",
#                          anchor0      = None,
#                          mass         = '2000.0 kg',
#                          centerOfMass = (0.5, '-1m', '0.8m'),
#                          boom         = ('3.0 m', '0deg', '0deg'),
#                          boomRng      = (None, None, ('0deg','360deg')))
#         boom1 = Boom( name         ='boom 1',
#                       description  = "The first boom. Can be lifted",
#                       anchor0      = pedestal,
#                       mass         = '200.0 kg',
#                       centerOfMass = (0.5, 0, 0),
#                       boom         = ('10.0 m', '90deg', '0deg'),
#                       boomRng      = (None, ('-90deg','90deg'), None))
#         boom2 = Boom( name         ='boom 2',
#                       description  = "The second boom. Can be lifted whole range",
#                       anchor0      = boom1,
#                       mass         = '100.0 kg',
#                       centerOfMass = (0.5, 0, 0),
#                       boom         = ('5.0 m', '90deg', '180deg'),
#                       boomRng      = (None, ('-180deg','180deg'), None))
#         rope = Boom( name         ='rope',
#                      description  = "The rope fixed to the last boom. Flexible connection",
#                      anchor0      = boom2,
#                      mass         = '50.0 kg', # so far basically the hook
#                      centerOfMass = 0.95, 
#                      boom         = ('1e-6 m', '180deg', '0 deg'),
#                      boomRng      = (('1e-6 m','20m'), ('90deg','270deg'), ('-180deg','180deg')),
#                      dampingQ     = 50.0,
#                      animationLW  = 2)
#         crane = Crane( name='TestCrane', description='A pedestal,2-booms,rope crane for testing', booms=[pedestal, boom1, boom2, rope])
#         return( crane)


if __name__ == '__main__':
#    suite = unittest.TestLoader().loadTestsFromTestCase(Test_WrapFMU) # use that to do all tests
    suite = unittest.TestSuite() # use this to load only single tests (together with next lines)"
    # single tests:
    suite.addTest( Test_Model("test_init"))
    test_result = unittest.TextTestRunner(verbosity=1).run(suite)
    if test_result.wasSuccessful():
        pass # possibility to perform cleanup when everything was successful
        #logger.info("Ran " +str( test_result.testsRun) +" tests. Success")
    else:
        print( test_result)

