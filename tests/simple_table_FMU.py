from component_model.model import Model
from component_model.component_FMUs import InputTable
from fmpy import dump, simulate_fmu, plot_result


class SimpleTable(InputTable):
    """This denotes the concrete table to turn into an FMU. Note: all parameters must be there to instantiate"""

    def __init__(self, **kwargs):
        print("SimpleTable init", kwargs)
        super().__init__(
            name="TestTable",
            description="my description",
            author="Siegfried Eisinger",
            version="0.1",
            table=((0.0, 1, 2, 3), (1.0, 4, 5, 6), (3.0, 7, 8, 9), (7.0, 8, 7, 6)),
            outputNames=("x", "y", "z"),
            interpolate=False,
            **kwargs,
        )


if __name__ == "__main__":

    def assertion(expr, txt):
        try:
            assert expr
        except:
            print(txt)

    mode = 3
    if mode in (0, 1):  # basic tests. Print out elements
        interpolate = False if mode == 0 else True
        tbl = InputTable(
            "TestTable",
            "my description",
            table=((0.0, 1, 2, 3), (1.0, 4, 5, 6), (3.0, 7, 8, 9), (7.0, 8, 8, 8)),
            outputNames=("x", "y", "z"),
            interpolate=interpolate,
        )
        assertion(
            tbl.interpolate.value == interpolate,
            f"Interpolation={interpolate} expected. Found {tbl.interpolate.value}",
        )
        assertion(
            all(tbl.times[i] == [0.0, 1.0, 3.0, 7.0][i] for i in range(tbl._rows)),
            f"Expected time array [0,1,3,7]. Found {tbl.times}",
        )
        for r in range(tbl._rows):
            assertion(
                all(
                    tbl.outputs[r][c]
                    == [[1, 2, 3], [4, 5, 6], [7, 8, 9], [8, 8, 8]][r][c]
                    for c in range(tbl._cols)
                ),
                f"Error in expected outputs, row {r}. Found {tbl.outputs[r]}",
            )
        assertion(
            all(tbl.ranges[0][c] == (1, 8)[c] for c in range(2)),
            f"Error in expected range of outputs, row 0. Found {tbl.ranges[0]}",
        )
        assertion(
            all(tbl.outs.value[c] == (1, 2, 3)[c] for c in range(tbl._cols)),
            f"Error in expected outs (row 0). Found {tbl.outs.value}",
        )
        tbl.setup_experiment(1.0)
        assertion(
            tbl.startTime == 1.0, f"Start time is expected 1.0. Found {tbl.startTime}"
        )
        tbl.enter_initialization_mode()  # iterate to row 1 (startTime)
        assertion(
            tbl.row == 1 and tbl.tInterval[0] == 1.0 and tbl.tInterval[1] == 3.0,
            f"Error in iterating to first row. Found ({tbl.row}, {tbl.tInterval})",
        )
        tbl.set_values(time=5.0)  # should iterate to row 2 and return interval (3,7)
        if interpolate:
            assertion(
                tbl.row == 2 and tbl.tInterval[0] == 3.0 and tbl.outs.value[0] == 7.5,
                f"Error in set_values(5.0). Found row {tbl.row}, interval {tbl.tInterval} => {tbl.outs.value}",
            )
        else:
            assertion(
                tbl.row == 2 and tbl.tInterval[0] == 3.0 and tbl.outs.value[0] == 7.0,
                f"Error in set_values(5.0). Found {tbl.outs.value}",
            )
        # print("Values(2)", tbl.get_values( 2.0, 0))
    elif mode in (2, 3):  # make FMU and run fmpy on the single component model
        interpolate = mode == 3
        asBuilt = Model.build(
            "simple_table_FMU.py", project_files=[]
        )  #'../component_model', ])
        result = simulate_fmu(
            asBuilt.name,
            stop_time=10.0,
            step_size=0.1,
            solver="Euler",
            debug_logging=True,
            logger=print,  # fmi_call_logger=print,
            start_values={"interpolate": interpolate},
        )
        print(dump(asBuilt.name))
        plot_result(result)
