import time
from pathlib import Path
from typing import Iterable

import pytest
from fmpy import plot_result, simulate_fmu  # type: ignore
from libcosimpy.CosimEnums import CosimErrorCode, CosimExecutionState
from libcosimpy.CosimExecution import CosimExecution
from libcosimpy.CosimLogging import CosimLogLevel, log_output_level
from libcosimpy.CosimManipulator import CosimManipulator
from libcosimpy.CosimObserver import CosimObserver
from libcosimpy.CosimSlave import CosimLocalSlave
from pythonfmu.fmi2slave import Fmi2Slave

from component_model.model import Model  # type: ignore
from component_model.utils.fmu import model_from_fmu


def match_par(txt: str, left: str = "(", right: str = ")"):
    pos0 = txt.find(left, 0)
    assert pos0 >= 0, f"First {left} not found"
    stack = [pos0]
    i = pos0
    while True:
        i += 1
        if len(txt) <= i:
            return (pos0, -1)
        elif txt[i] == "#":  # comment
            i = txt.find("\n", i)
        elif txt[i:].startswith(left):
            stack.append(i)
        elif txt[i:].startswith(right):
            if len(stack) > 1:
                stack.pop(-1)
            else:
                return (pos0, i)


def test_match_par():
    txt = "Hello (World)"
    res = match_par(txt)
    assert res == (6, 12)
    assert txt[res[0]] == "("
    assert txt[res[1]] == ")"
    assert txt[res[0] + 1 : res[1]] == "World"
    txt = "def  __init__( x:float=0.5, tpl:tuple=(1,2,3), tpl2:tuple=(1,2,(11,12))):\n code"
    res = match_par(txt)
    assert txt[res[0] :].startswith("( x:float")
    assert txt[: res[1] + 1].endswith("12)))")
    txt = "def  __init__( x:float=0.5,\n tpl:tuple=(1,2,3), #a (stupid((comment)\n tpl2:tuple=(1,2,(11,12)),\n):\n code"
    res = match_par(txt)
    assert txt[res[0] :].startswith("( x:float")
    assert txt[: res[1] + 1].endswith("12)),\n)")


def model_parameters(src: Path, newargs: dict | None = None) -> tuple[str, Fmi2Slave]:
    """Replace default parameters in model class __init__ and return adapted script as str.
    Function checks also that a unique FMI2Slave class exists in the script.

    Args:
        src (Path): Path to the module source file where the class definition is expected.
        newargs (dict): Optional dict of new argument values provided as name : value
           If not provided, the script is returned unchanged as str
    Returns:
        (adapted) script as str, model class object
    """
    import importlib
    import inspect
    import sys

    modulename = src.stem
    if src.parent not in sys.path:
        sys.path.insert(0, str(src.parent))
    module = importlib.import_module(modulename)
    src_file = inspect.getsourcefile(module)
    assert src_file is not None and Path(src_file) == src
    assert inspect.ismodule(module)
    modelclasses = {}
    for _, obj in inspect.getmembers(module):
        if inspect.isclass(obj):
            mro = inspect.getmro(obj)
            if Fmi2Slave in mro and not inspect.isabstract(obj):
                modelclasses.update({obj: len(mro)})
    if not len(modelclasses):
        raise ValueError(f"No child class of Fmi2Slave found in module {src}") from None
    else:
        model = None
        init = None
        maxlen = max(n for n in modelclasses.values())
        classes = [c for c, n in modelclasses.items() if n == maxlen]
        if not len(classes):
            raise ValueError(f"No child class of Fmi2Slave found in module {src}") from None
        elif len(classes) > 1:
            raise ValueError(f"Non-unique Fmi2Slave-derived class in module {src}. Found {classes}.") from None
        else:
            model = classes[0]
            for name, obj in inspect.getmembers(model):
                if inspect.isfunction(obj) and name == "__init__":
                    init = obj
                    break
    assert model is not None, f"Model object not found in module {src}"
    module_lines = inspect.getsourcelines(module)

    if newargs is None:  # just return the raw script as str
        return ("".join(line for line in module_lines[0]), model)

    assert init is not None, f"__init__() function not found in module {src}, model {model}"
    sig = inspect.signature(init)
    pars = sig.parameters
    newpars = []
    for p in pars:
        if p in newargs:
            par = pars[p].replace(default=newargs[p])
        else:
            par = pars[p]
        newpars.append(par)
        signew = inspect.Signature(parameters=newpars)
    init_line = inspect.getsourcelines(init)[1]
    from_init = "".join(line for line in module_lines[0][init_line - 1 :])
    init_pos = from_init.find("__init__")
    start, end = (match_par(from_init[init_pos - 1 :])[i] + init_pos for i in range(2))
    from_init = from_init.replace(from_init[start - 1 : end], str(signew), 1)
    module_code = "".join(line for line in module_lines[0][: init_line - 1]) + from_init
    return (module_code, model)


def test_model_parameters():
    with pytest.raises(ValueError) as err:
        model_parameters(Path(__file__).parent / "examples" / "new_pythonfmu_features3.py", {})
    assert str(err.value).startswith("Non-unique Fmi2Slave-derived class in module")
    with pytest.raises(ValueError) as err:
        model_parameters(Path(__file__).parent / "examples" / "new_pythonfmu_features4.py", {})
    assert str(err.value).startswith("No child class of Fmi2Slave found in module")
    module_code, model = model_parameters(
        Path(__file__).parent / "examples" / "new_pythonfmu_features2.py", newargs={"i": 7, "f": -9.9, "s": "World"}
    )
    with open("test.py", "w") as f:
        f.write(module_code)
    module_code, model = model_parameters(
        Path(__file__).parent / "examples" / "new_pythonfmu_features.py", newargs={"i": 7, "f": -9.9, "s": "World"}
    )


def build_fmu(
    model: str | Path,
    dest: str | Path = ".",
    project_files: Iterable[str | Path] = set(),
    documentation_folder: str | Path | None = None,
    newargs: dict | None = None,
    **options,
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
    # Replace "shutil.copy2(script_file, temp_dir)" code line
    if newargs is not None:  # default arguments to be replaced
        new_script, model_class = model_parameters(Path(model), newargs)
        # Change that so that it points to the temp_dir
        with open(model_class.name, "w") as f:
            f.write(new_script)
    else:
        # Here we use the original shutil.copy2()
        pass

    # Alternatively, 'model_parameters' can be used on the top of the function
    # (performing checks and returning the script as str, whether newargs are provided or not)
    # The shutil.copy2() function can then be replaced with writing the str to file at the temp folder.


def test_adapted():
    src = Path(__file__).parent / "examples" / "new_pythonfmu_features.py"
    build_fmu(src, newargs={"i": 8, "f": -9.91, "s": "World2"})


@pytest.fixture(scope="session")
def new_features_fmu():
    return _new_features_fmu()


def _new_features_fmu():
    build_path = Path.cwd()
    build_path.mkdir(exist_ok=True)
    src = Path(__file__).parent / "examples" / "new_pythonfmu_features2.py"
    _src2 = build_fmu(str(src) + ".NewFeatures", newargs={"i": 9, "f": -9.92, "s": "World3"})
    src2 = Path(__file__).parent / "test_working_directory" / "new_pythonfmu_features2.py"
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
    assert isinstance(nf, Model)


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
            "i": 0,
            "f": 99.9,
            "s": "World",
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

    log_output_level(CosimLogLevel.TRACE)
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
    assert variables == {"i": 0, "f": 1, "s": 2}
    # Set initial values. Actual setting will only happen after start_initialization_mode
    sim.real_initial_value(_nf, variables["f"], 99.9)  #
    sim.string_initial_value(_nf, variables["s"], "World")

    assert get_status(sim)["state"] == "STOPPED"

    observer = CosimObserver.create_last_value()
    assert sim.add_observer(observer)
    manipulator = CosimManipulator.create_override()
    assert sim.add_manipulator(manipulator)

    print("Simulate step 1. Log level TRACE : logAll")
    assert sim.simulate_until(target_time=1e9), "Simulate for one base step did not work"
    # print(f"STATUS {get_status(sim)}")
    assert get_status(sim)["currentTime"] == 1e9, "Time after simulation not correct"
    assert observer.last_real_values(0, [variables["f"]])[0] == 0.0, "The current time at step"
    assert observer.last_string_values(0, [variables["s"]])[0].decode() == "World"

    log_output_level(CosimLogLevel.DEBUG)
    print("Simulate step 2. Log level DEBUG : -= ok")
    assert sim.simulate_until(target_time=2e9), "Simulate for one base step did not work"
    assert observer.last_real_values(0, [variables["f"]])[0] == 1.0, "The current time at step 2"

    log_output_level(CosimLogLevel.INFO)
    print("Simulate step 3. Log level INFO : -= discard")
    assert sim.simulate_until(target_time=3e9), "Simulate for one base step did not work"

    log_output_level(CosimLogLevel.WARNING)
    print("Simulate step 4. Log level WARNING : == INFO")
    assert sim.simulate_until(target_time=4e9), "Simulate for one base step did not work"

    log_output_level(CosimLogLevel.ERROR)
    print("Simulate step 5. Log level ERROR : -= warning")
    assert sim.simulate_until(target_time=5e9), "Simulate for one base step did not work"

    log_output_level(CosimLogLevel.FATAL)
    print("Simulate step 6. Log level FATAL : nothing??")
    assert sim.simulate_until(target_time=6e9), "Simulate for one base step did not work"
    assert observer.last_real_values(0, [variables["f"]])[0] == 5.0, "The current time at step 6"

    log_output_level(CosimLogLevel.ERROR)
    print("Simulate steps >6. Log level ERROR. terminate() after 6")
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
    assert model["copyright"] == f"Copyright (c) {time.localtime()[0]} Siegfried Eisinger", (
        f"Found: {model['copyright']}"
    )
    de = model["default_experiment"]
    assert de is not None
    assert de["start_time"] == 0.0
    assert de["step_size"] == 1.0
    assert de["stop_time"] == 9.0


if __name__ == "__main__":
    retcode = pytest.main(["-rA", "-v", "--rootdir", "../", "--show", "False", __file__])
    assert retcode == 0, f"Non-zero return code {retcode}"

    # test_new_features_class()
    # import os

    # os.chdir(Path(__file__).parent.absolute() / "test_working_directory")
    # test_model_parameters()
    # test_match_par()
    # new = _plain_fmu()
    # adapted = _new_features_fmu()
    # test_use_fmu( new, show=False)
    # test_from_fmu(new)
    # test_from_osp(new)
    # test_from_osp(adapted)
    # test_adapted()
