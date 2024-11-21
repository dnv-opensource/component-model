import time
import xml.etree.ElementTree as ET  # noqa: N817
from math import sqrt
from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
import pytest
from pythonfmu.fmi2slave import Fmi2Slave
from fmpy import plot_result, simulate_fmu  # type: ignore
from fmpy.util import fmu_info  # type: ignore
from fmpy.validation import validate_fmu  # type: ignore
from libcosimpy.CosimLogging import CosimLogLevel, log_output_level
from libcosimpy.CosimEnums import CosimErrorCode, CosimExecutionState
from libcosimpy.CosimExecution import CosimExecution
from libcosimpy.CosimManipulator import CosimManipulator
from libcosimpy.CosimObserver import CosimObserver
from libcosimpy.CosimSlave import CosimLocalSlave
from typing import Iterable
from component_model.model import Model  # type: ignore
from component_model.utils.fmu import model_from_fmu


def _in_interval(x: float, x0: float, x1: float):
    return x0 <= x <= x1 or x1 <= x <= x0


def _to_et(file: str, sub: str = "modelDescription.xml"):
    with ZipFile(file) as zp:
        xml = zp.read(sub)
    return ET.fromstring(xml)

# def replace_assert( txt:str):
#     c = re.compile(r"assert\s([^,]+),([^\n]*)\n")
#     pos = 0
#     while True:
#         m = c.search( txt[pos:])
#         if m is not None:
#             print(m.groups())
#             pos += m.end()
#         else:
#             break
    

def build_FMU(
        model: str | Path,
        dest: str | Path = ".",
        project_files: Iterable[str|Path] = set(),
        documentation_folder: str | Path | None = None,
        newargs:dict|None=None,
        **options
        ):
    """PythonFMU build function with additional argument to replace default argument values

    Args:
        model (str | Path): model object identificator replacing script_file. Several possibilities:
        
           * str | Path pointing to an existing python script file (as in PythonFMU version <= 0.6.5)
              A check is performed that exacly one Fmi2Slave-derived class resides in the file
              and this class is then used to build FMU
           * str denoting a python module.class derived from Fmi2Slave.
              The calling code must either ensure that this python class is in the python path,
              or the path can be prepended like src-file-path(-with.py).class-name
        dest, project_files, documentation_folder: As in FMUBuilder.build_fmu
        newargs (dict): Optional new default values of class __init__ parameters.
           All keys in this dict must correspond to parameter names of the class
           and the values must have the correct type.
        **options: As in FMUBuilder.build_fmu        
    """
    from abc import ABCMeta
    import importlib
    import inspect
    import re

    def get_src_module_cls( model:str|Path):
        """Identify the script file and the class object"""
        import sys
        if isinstance( model, str) and not Path(model).exists(): # assume module-name.class-name
            assert '.' in model, f"Module information missing in {model}"
            classname = model.split('.')[-1]
            path = model[:-len(classname)-1]
            if not path.endswith('.py'):
                path += '.py'
            src = Path( path)
                
        else: # must be a script identificator
            src = Path( model)
            classname = ""

        assert src.exists(), f"Model source file within model identificator {model} not found"

        modulename = src.name[:-3]
        if src.parent not in sys.path:
            sys.path.insert(0, str(src.parent))
        module = importlib.import_module(modulename)
        if classname=="": # class not yet identified
            classobj = None
            for name, cls in inspect.getmembers( module, inspect.isclass):
                if cls.__module__ == modulename and Fmi2Slave in cls.mro():
                    if classobj is None:
                        classobj = cls
                    else:
                        raise ValueError(f"Several Fmi2Slave classes in file {src}. Fully qualify class!") from None
            assert classobj is not None, f"No Fmi2Slave class found in file {src}"
        else:
            classobj = getattr( module, classname)

        return (src, module, classobj)
    
    def replace_values( src:Path, classobj: ABCMeta, newargs:dict):
        """In src, classobj.__init__ replace the argument default values as given in newargs.
        Return the src file as changed str.
        """
        assert isinstance( classobj, ABCMeta)
        sig = inspect.signature(classobj.__init__)
        pars = sig.parameters
        newpars = []
        for p in pars:
            if p in newargs:
                par = pars[p].replace( default=newargs[p])
            else:
                par = pars[p]
            newpars.append( par)
        signew = inspect.Signature(parameters=newpars)
        #print(str(signew), classobj.__name__)
        with open(src) as f:
            _src = f.read()
        m = re.search(r"class\s+"+re.escape(classobj.__name__)+"\s*\(", _src)
        changed = _src[:m.end()] + re.sub(r"def\s+__init__\s*(\(.+\))", "def __init__"+str(signew), _src[m.end():])
        return changed
        
    # Replace the first 5 code lines with this line:
    script_file, module, classobj = get_src_module_cls( model)

    # Replace "shutil.copy2(script_file, temp_dir)" code line 
    if newargs is not None: # default arguments to be replaced
        new_script = replace_values( script_file, classobj, newargs)
        # Change that so that it points to the temp_dir
        with open(script_file.name, 'w') as f:
            f.write( new_script)
    else:
        # Here we use the original shutil.copy2()
        pass


def test_adapted():
    src = Path(__file__).parent / "examples" / "new_pythonfmu_features.py"
    build_FMU( str(src)+".NewFeatures", newargs = {'i':7, 'f':-9.9, 's':"World"}) # fully qualified
    build_FMU( src, newargs = {'i':8, 'f':-9.91, 's':"World2"}) # only script
    


@pytest.fixture(scope="session")
def new_features_fmu():
    return _new_features_fmu()

def _new_features_fmu():
    build_path = Path.cwd()
    build_path.mkdir(exist_ok=True)
    src = Path(__file__).parent / "examples" / "new_pythonfmu_features.py"
    _src2 = build_FMU( str(src)+".NewFeatures", newargs = {'i':9, 'f':-9.92, 's':"World3"})
    src2 = Path(__file__).parent / "test_working_directory" / "new_pythonfmu_features.py"
    fmu_path = Model.build(
        src2,
        project_files=[],
        dest=build_path,
    )
    return fmu_path



@pytest.fixture(scope="session")
def plain_fmu():
    return _plain_fmu()

def _plain_fmu():
    build_path = Path.cwd()
    build_path.mkdir(exist_ok=True)
    fmu_path = Model.build(
        str(Path(__file__).parent / "examples" / "new_pythonfmu_features.py"),
        project_files=[],
        dest=build_path,
    )
    return fmu_path
    

def test_new_features_class():
    """Test the BouncingBall3D class in isolation.

    The first four lines are necessary to ensure that the BouncingBall3D class can be accessed:
    If pytest is run from the command line, the current directory is the package root,
    but when it is run from the editor (__main__) it is run from /tests/.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).parent / "examples"))
    from new_pythonfmu_features import NewFeatures  # type: ignore

    nf = NewFeatures()


def test_use_fmu(plain_fmu, show):
    """Test and validate the NewFeatures using fmpy and not using OSP."""
    assert plain_fmu.exists(), f"File {plain_fmu} does not exist"
    dt = 1
    result = simulate_fmu(
        plain_fmu,
        start_time=0.0,
        stop_time=9.0,
        step_size=dt,
        validate=True,
        solver="Euler",
        debug_logging=True,
        visible=True,
        logger=print,  # fmi_call_logger=print,
        start_values={
            "i" : 0,
            "f" : 99.9,
            "s" : "World",
        },
    )
    if show:
        plot_result(result)


def test_from_osp(plain_fmu):
    def get_status(sim):
        status = sim.status()
        return {
            "currentTime": status.current_time,
            "state": CosimExecutionState(status.state).name,
            "error_code": CosimErrorCode(status.error_code).name,
            "real_time_factor": status.real_time_factor,
            "rolling_average_real_time_factor": status.rolling_average_real_time_factor,
            "real_time_factor_target": status.real_time_factor_target,
            "is_real_time_simulation": status.is_real_time_simulation,
            "steps_to_monitor": status.steps_to_monitor,
        }
    log_output_level( CosimLogLevel.TRACE)
    sim = CosimExecution.from_step_size(step_size=1e9)  # empty execution object with fixed time step in nanos
    nf = CosimLocalSlave(fmu_path=str(plain_fmu.absolute()), instance_name="nf")

    _nf = sim.add_local_slave(nf)
    assert _nf == 0, f"local slave number {_nf}"
    info = sim.slave_infos()
    assert info[0].name.decode() == "nf", "The name of the component instance"
    assert info[0].index == 0, "The index of the component instance"
    assert sim.slave_index_from_instance_name("nf") == 0
    assert sim.num_slaves() == 1
    assert sim.num_slave_variables(0) == 3
    variables = {var_ref.name.decode(): var_ref.reference for var_ref in sim.slave_variables(_nf)}
    assert variables == {
        "i": 0,
        "f": 1,
        "s": 2
    }
    # Set initial values. Actual setting will only happen after start_initialization_mode
    sim.real_initial_value(_nf, variables["f"], 99.9)  # 
    sim.string_initial_value(_nf, variables["s"], "World")

    assert get_status(sim)["state"] == "STOPPED"

    observer = CosimObserver.create_last_value()
    assert sim.add_observer(observer)
    manipulator = CosimManipulator.create_override()
    assert sim.add_manipulator(manipulator)

    print(f"Simulate step 1. Log level TRACE : logAll")
    assert sim.simulate_until(target_time=1e9), "Simulate for one base step did not work"
    #print(f"STATUS {get_status(sim)}")
    assert get_status(sim)["currentTime"] == 1e9, "Time after simulation not correct"
    assert observer.last_real_values(0, [variables['f']])[0] == 0.0, "The current time at step"
    assert observer.last_string_values(0, [variables['s']])[0].decode() == "World"
    
    log_output_level( CosimLogLevel.DEBUG)
    print(f"Simulate step 2. Log level DEBUG : -= ok")
    assert sim.simulate_until(target_time=2e9), "Simulate for one base step did not work"
    assert observer.last_real_values(0, [variables['f']])[0] == 1.0, "The current time at step 2"

    log_output_level( CosimLogLevel.INFO)
    print(f"Simulate step 3. Log level INFO : -= discard")
    assert sim.simulate_until(target_time=3e9), "Simulate for one base step did not work"

    
    log_output_level( CosimLogLevel.WARNING)
    print(f"Simulate step 4. Log level WARNING : == INFO")
    assert sim.simulate_until(target_time=4e9), "Simulate for one base step did not work"
    
    log_output_level( CosimLogLevel.ERROR)
    print(f"Simulate step 5. Log level ERROR : -= warning")
    assert sim.simulate_until(target_time=5e9), "Simulate for one base step did not work"
    
    log_output_level( CosimLogLevel.FATAL)
    print(f"Simulate step 6. Log level FATAL : nothing??")
    assert sim.simulate_until(target_time=6e9), "Simulate for one base step did not work"
    assert observer.last_real_values(0, [variables['f']])[0] == 5.0, "The current time at step 6"    

    log_output_level( CosimLogLevel.ERROR)
    print(f"Simulate steps >6. Log level ERROR. terminate() after 6")
    assert sim.simulate_until(target_time=7e9), "Simulate for one base step did not work"
    assert sim.simulate_until(target_time=8e9), "Simulate for one base step did not work"
    assert sim.simulate_until(target_time=9e9), "Simulate for one base step did not work"


def test_from_fmu(plain_fmu):
    assert plain_fmu.exists(), "FMU not found"
    model = model_from_fmu(plain_fmu)
    assert model["name"] == "NewFeatures", f"Name: {model['name']}"
    assert model["description"] == "Dummy model for testing new features in PythonFMU"
    assert model["author"] == "Siegfried Eisinger"
    assert model["version"] == "0.1"
    assert model["license"].startswith("Permission is hereby granted, free of charge, to any person obtaining a copy")
    assert model["copyright"] == f"Copyright (c) {time.localtime()[0]} Siegfried Eisinger", f"Found: {model.copyright}"
    assert model["default_experiment"] is not None
    assert (
        model["default_experiment"]["start_time"],
        model["default_experiment"]["step_size"],
        model["default_experiment"]["stop_time"],
    ) == (0.0, 0.01, 1.0)


if __name__ == "__main__":
    # retcode = pytest.main(["-rA", "-v", "--rootdir", "../", "--show", "False", __file__])
    # assert retcode == 0, f"Non-zero return code {retcode}"
    # test_new_features_class()
    import os
    os.chdir(Path(__file__).parent.absolute() / "test_working_directory")
    new = _plain_fmu()
    adapted = _new_features_fmu()
    # test_use_fmu( new, show=False)
    # test_from_fmu( new)
    # test_from_osp(new)
    test_from_osp(adapted)
    # test_adapted()
