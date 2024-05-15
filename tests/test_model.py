import logging
from pathlib import Path
import time

from component_model.logger import get_module_logger
from component_model.model import Model, make_osp_system_structure, model_from_fmu

logger = get_module_logger(__name__, level=logging.INFO)


def test_license():
    mod = Model("TestModel", author="Ola Norman")
    c, lic = mod.make_copyright_license(None, None)
    assert c == f"Copyright (c) {time.localtime()[0]} Ola Norman", f"Found: {c}"
    assert lic.startswith("Permission is hereby granted, free of charge, to any person obtaining a copy")
    _lic = """Copyright (c) 2023 Ola Norman

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and a ..."""
    c, lic = mod.make_copyright_license(None, _lic)
    assert c == "Copyright (c) 2023 Ola Norman"
    assert lic.startswith("Permission is hereby granted, free of charge, to any person obtaining a copy")

    c, lic = mod.make_copyright_license("Copyleft (c) 3000 Nobody", _lic)
    assert c == "Copyleft (c) 3000 Nobody"
    assert lic.strip().startswith("Permission is hereby granted, free of charge, to any person obtaining a copy")


#    def test_model_description(self):
def test_osp_structure():
    make_osp_system_structure(
        "systemModel",
        version="0.1",
        models={
            "simpleTable": {"interpolate": True},
            "mobileCrane": {"pedestal.pedestalMass": 5000.0, "boom.boom.0": 20.0},
        },
        connections=("simpleTable", "outputs.0", "mobileCrane", "pedestal.angularVelocity"),
    )


# def test_from_fmu():
#     path = Path( Path(__file__).parent, "BouncingBallFMU.fmu")
#     assert path.exists(), "FMU not found"
#     model = model_from_fmu( path)
#     assert model.name == "BouncingBallFMU", f"Name:{model.name}"
#     print( dir(model))
#     assert model.description == "Simple bouncing ball test FMU", f"Description:{model.description}"
#     assert model.author == "DNV, SEACo project"
#     assert model.version == "0.1"
#     assert model.license.startswith("Permission is hereby granted, free of charge, to any person obtaining a copy")
#     assert model.copyright == f"Copyright (c) {time.localtime()[0]} DNV, SEACo project", f"Found: {model.copyright}"
#     assert model.guid == "8336c04d3e2f45379db8ed685e034a69", f"Found: {model.guid}"
#     assert model.default_experiment is None
#     #     assert (
#     #         model.default_experiment.start_time,
#     #         model.default_experiment.step_size,
#     #         model.default_experiment.stop_time,
#     #         model.default_experiment.tolerance,
#     #     ) == (0.0, 0.1, 10.0, 0.001)
#     assert model.flags == {
#         "needsExecutionTool": True,
#         "canHandleVariableCommunicationStepSize": True,
#         "canNotUseMemoryManagementFunctions": True,
#     }
#     for idx, var in model.vars.items():
#         print(idx, var)
#     assert model.vars[0].name == "x[0]"
#     assert model.vars[0].value0 == 0.0
#     assert model.vars[6].name == "bounceFactor"


if __name__ == "__main__":
    test_license()
    test_osp_structure()
#    test_from_fmu() # not yet finished and tested
