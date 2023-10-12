import sys, os
import math
import numpy as np
sys.path.insert(0, os.path.abspath('../'))
import logging
from component_model.logger import get_module_logger
logger = get_module_logger(__name__, level=logging.INFO)

from component_model.model import Model, ModelInitError

import unittest


class Test_model( unittest.TestCase):
    """ Test suite for the variable module """                         

    def test_license(self):
        mod = Model("TestModel", author="Ola Norman")
        print("AU", mod.author)
        c, l = mod.make_copyright_license( None, None)
        self.assertEqual( c, 'Copyright (c) 2023 Ola Norman')
        self.assertTrue( l.startswith("Permission is hereby granted, free of charge, to any person obtaining a copy\nof this software"))
        lic = '''Copyright (c) 2023 Ola Norman

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and a ...'''
        c,l = mod.make_copyright_license( None, lic)
        self.assertEqual( c, 'Copyright (c) 2023 Ola Norman')
        self.assertTrue( l.startswith("Permission is hereby granted, free of charge, to any person obtaining a copy\nof this software and a .."))

        c,l = mod.make_copyright_license( "Copyleft (c) 3000 Nobody", lic)
        self.assertEqual( c, "Copyleft (c) 3000 Nobody")
        self.assertTrue( l.startswith("Permission is hereby granted, free of charge, to any person obtaining a copy\nof this software and a .."))

    def test_model_description(self):


if __name__ == '__main__':
#    suite = unittest.TestLoader().loadTestsFromTestCase(Test_model) # use that to do all tests
    suite = unittest.TestSuite() # use this to load only single tests (together with next lines)
    # single tests:
    suite.addTest( Test_model("test_license")) 
    test_result = unittest.TextTestRunner(verbosity=1).run(suite)
    if test_result.wasSuccessful():
        pass # possibility to perform cleanup when everything was successful
        #logger.info("Ran " +str( test_result.testsRun) +" tests. Success")
    else:
        print( test_result)

