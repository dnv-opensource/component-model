from pathlib import Path

import pytest
from component_model.example_models.bouncing_ball_xz import BouncingBallXZ
from component_model.model import Model  # type: ignore
from fmpy import simulate_fmu  # type: ignore
from libcosimpy.CosimEnums import CosimExecutionState
from libcosimpy.CosimExecution import CosimExecution
from libcosimpy.CosimSlave import CosimLocalSlave


@pytest.fixture(scope="session")
def bouncing_ball_fmu():
    build_path = Path.cwd() / "fmus"
    build_path.mkdir(exist_ok=True)
    fmu_path = Model.build(
        str(Path(__file__).parent.parent / "component_model" / "example_models" / "bouncing_ball_xz.py"),
        project_files=[],
        dest=build_path,
    )
    return fmu_path


def test_model_description():
    mod = BouncingBallXZ()
    mod.to_xml()


def test_run_fmpy(bouncing_ball_fmu):
    _ = simulate_fmu(
        bouncing_ball_fmu,
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


def test_run_osp(bouncing_ball_fmu):
    sim = CosimExecution.from_step_size(step_size=1e8)  # empty execution object with fixed time step in nanos
    bb = CosimLocalSlave(fmu_path=(str(bouncing_ball_fmu.absolute())), instance_name="bb")
    print("SLAVE", bb, sim.status())

    ibb = sim.add_local_slave(bb)
    assert ibb == 0, f"local slave number {ibb}"
    sim_status = sim.status()
    assert sim_status.current_time == 0
    assert CosimExecutionState(sim_status.state) == CosimExecutionState.STOPPED
    infos = sim.slave_infos()
    print("INFOS", infos)
    reference_dict = {var_ref.name.decode(): var_ref.reference for var_ref in sim.slave_variables(ibb)}
    # Set initial values
    sim.real_initial_value(ibb, reference_dict["v0[0]"], 2.0)
    sim.real_initial_value(ibb, reference_dict["v0[1]"], 3.0)

    # Simulate for 15 seconds
    sim.simulate_until(target_time=15e9)


if __name__ == "__main__":
    retcode = pytest.main(["-rA", "-v", __file__])
    assert retcode == 0, f"Non-zero return code {retcode}"
