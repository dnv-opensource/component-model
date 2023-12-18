import sys, os
import math
import numpy as np

sys.path.insert(0, os.path.abspath("../"))
import logging
from component_model.logger import get_module_logger

logger = get_module_logger(__name__, level=logging.INFO)

from component_model.model import Model, ModelInitError, make_OSP_system_structure, model_from_fmu

import unittest


class Test_model(unittest.TestCase):
    """Test suite for the variable module"""

    def test_license(self):
        mod = Model("TestModel", author="Ola Norman")
        print("AU", mod.author)
        c, l = mod.make_copyright_license(None, None)
        self.assertEqual(c, "Copyright (c) 2023 Ola Norman")
        self.assertTrue(
            l.startswith(
                "Permission is hereby granted, free of charge, to any person obtaining a copy\nof this software"
            )
        )
        lic = """Copyright (c) 2023 Ola Norman

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and a ..."""
        c, l = mod.make_copyright_license(None, lic)
        self.assertEqual(c, "Copyright (c) 2023 Ola Norman")
        self.assertTrue(
            l.startswith(
                "Permission is hereby granted, free of charge, to any person obtaining a copy\nof this software and a .."
            )
        )

        c, l = mod.make_copyright_license("Copyleft (c) 3000 Nobody", lic)
        self.assertEqual(c, "Copyleft (c) 3000 Nobody")
        self.assertTrue(
            l.startswith(
                "Permission is hereby granted, free of charge, to any person obtaining a copy\nof this software and a .."
            )
        )

    #    def test_model_description(self):
    def test_osp_structure(self):
        make_OSP_system_structure(
            "systemModel",
            version="0.1",
            models={
                "simpleTable": {"interpolate": True},
                "mobileCrane": {"pedestal.pedestalMass": 5000.0, "boom.boom.0": 20.0},
            },
            connections=(
                ("simpleTable", "outputs.0", "mobileCrane", "pedestal.angularVelocity"),
            ),
        )
    def test_from_fmu(self):
        model = model_from_fmu( 'BouncingBallFMU.fmu' )
        self.assertEqual( model.name, "BouncingBallFMU")
        self.assertEqual( model.description, "Simple bouncing ball test FMU")
        self.assertEqual( model.author, "DNV, SEACo project"),
        self.assertEqual( model.version, "0.1"),
        self.assertTrue( model.license.startswith( "Permission is hereby granted, free of charge, to any person obtaining a copy")),
        self.assertEqual( model.copyright, "Copyright (c) 2023 DNV, SEACo project"),
        self.assertEqual( model.guid, "06128d688f4d404d8f6d49d6e493946b"),
        self.assertEqual( (model.default_experiment.start_time, model.default_experiment.step_size, model.default_experiment.stop_time, model.default_experiment.tolerance),
                          (0.0, 0.1, 10.0, 0.001))
        self.assertEqual( model.nonDefaultFlags, {'needsExecutionTool': True, 'canHandleVariableCommunicationStepSize': True, 'canNotUseMemoryManagementFunctions': True})
        for idx, var in model.vars.items():
            print( idx, var)
        self.assertEqual( model.vars[0].name, 'x')
        self.assertEqual( model.vars[0].initialVal, (0.0, 0.0))
        self.assertEqual( model.vars[6].name, 'energy')



if __name__ == "__main__":
    #    suite = unittest.TestLoader().loadTestsFromTestCase(Test_model) # use that to do all tests
    suite = (
        unittest.TestSuite()
    )  # use this to load only single tests (together with next lines)
    # single tests:
#    suite.addTest( Test_model("test_license"))
#    suite.addTest(Test_model("test_osp_structure"))
    suite.addTest(Test_model("test_from_fmu"))
    test_result = unittest.TextTestRunner(verbosity=1).run(suite)
    if test_result.wasSuccessful():
        pass  # possibility to perform cleanup when everything was successful
        # logger.info("Ran " +str( test_result.testsRun) +" tests. Success")
    else:
        print(test_result)
