from math import sqrt

import numpy as np
from component_model.logger import get_module_logger
from component_model.model import Model
from component_model.variable import Variable, VariableNP
from fmpy import dump, plot_result, simulate_fmu  # type: ignore
from libcosimpy.CosimExecution import CosimExecution
from libcosimpy.CosimSlave import CosimLocalSlave
from libcosimpy.CosimEnums import CosimExecutionState
from models.BouncingBall import BouncingBallFMU

def test_model_description():
    mod = BouncingBallFMU()
    mod.to_xml()


def test_make_fmu():
    built = Model.build("./models/BouncingBall.py")
    dump(built)
    return built


def test_run_fmpy():
    result = simulate_fmu(
        "BouncingBallFMU.fmu",
        start_time=0.0,
        stop_time=6.5,  # 10.0,
        step_size=0.1,
        validate=True,
        solver="Euler",
        debug_logging=False,
        visible=True,
        logger=print,  # fmi_call_logger=print,
        start_values={
            "v0[0]": 2.0,
            "v0[1]": 3.0,
        },
    )
    plot_result(result)


def test_run_osp():
    sim = CosimExecution.from_step_size(step_size=1e8)  # empty execution object with fixed time step in nanos
    bb = CosimLocalSlave(fmu_path="./BouncingBallFMU.fmu", instance_name="bb")
    print("SLAVE", bb, sim.status())


    ibb = sim.add_local_slave(bb)
    assert ibb==0, f"local slave number {ibb}"
    sim_status = sim.status()
    assert sim_status.current_time==0
    assert CosimExecutionState(sim_status.state) == CosimExecutionState.STOPPED
    infos = sim.slave_infos()
    print("INFOS", infos)
    reference_dict = {
        var_ref.name.decode(): var_ref.reference for var_ref in sim.slave_variables(ibb)
    }
    #Set initial values
    sim.real_initial_value(ibb, reference_dict["v0[0]"], 2.0)
    sim.real_initial_value(ibb, reference_dict["v0[1]"], 3.0)

    #Simulate for 15 seconds
    sim.simulate_until(target_time=15e9)

if __name__ == "__main__":
    #logger = get_module_logger(__name__, level=1)  # 0:>0Warning, 1:all,
    test_model_description()
    test_make_fmu()
    test_run_fmpy()
    test_run_osp()
