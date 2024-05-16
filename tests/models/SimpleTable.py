from component_model.component_fmus import InputTable  # Note: needed even if only running the FMU!


class SimpleTable(InputTable):
    """This denotes the concrete table turned into an FMU.
    Exposed interface variables are 'outs' (array) and 'interpolate'"""

    def __init__(self, **kwargs):
        #        print("SimpleTable init", kwargs)
        super().__init__(
            name="TestTable",
            description="my description",
            author="Siegfried Eisinger",
            version="0.1",
            table=((0.0, 1, 2, 3), (1.0, 4, 5, 6), (3.0, 7, 8, 9), (7.0, 8, 7, 6)),
            defaultExperiment={"start_time": 0.0, "stop_time": 10.0, "step_size": 0.1},
            outputNames=("x", "y", "z"),
            interpolate=False,
            **kwargs,
        )
