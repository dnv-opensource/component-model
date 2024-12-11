from component_model.model import Model
from pathlib import Path

def test_build_custom_params():
    build_path = Path.cwd()
    build_path.mkdir(exist_ok=True)
    fmu_path = Model.build(
        Path(__file__).parent / "examples" / "bouncing_ball_3d.py",
        project_files=[],
        dest=build_path,
        newargs={
            "speed": ("0.5 m/s", "2.0 m/s", "2 m/s"),
            "e": 1.5,
        }
    )

