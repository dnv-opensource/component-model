import time
from dataclasses import dataclass
from enum import Enum
from multiprocessing import Process, Queue
from typing import Any, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from libcosimpy.CosimExecution import CosimExecution  # type: ignore
from libcosimpy.CosimObserver import CosimObserver  # type: ignore
from mpl_toolkits.mplot3d.axes3d import Axes3D, Line3D  # type: ignore


class SimulatorStatus(Enum):
    stopped = 0
    stepping = 1
    completed = 2


# Used to store more metadata about signals so that it's easier to fetch time-series data from the simulator
@dataclass
class Variable:
    name: str
    instance: str
    value_reference: int = 0
    slave_index: int = 0

    def __init__(self, **entries: Any):
        """Variable constructor."""
        self.__dict__.update(entries)

    def id(self):
        return f"{self.instance}.{self.name}"

    @staticmethod
    def get_id(_: "Variable", instance: str, name: str):
        return f"{instance}.{name}"


# OSPSignal type := A tuple consisting of instance name (OSP simulator name) and the name of the output port
OSPSignal = Tuple[str, str]


class VisualSimulator:
    """
    Visual simulator allows to plot out signals from a real-time OSP simulation in a 3D matplotlib instance.
    The implementations consists of two different processes that communicate with each other, one that runs
    the simulation using libcosimpy and another one that's constantly re-rendering the 3d points in the plot.

    Note: Interactions with the matplotlib interface may block the rendering which will also block the simulation.
    This is because matplotlib can't handle the rendering a of a new observer position and the new 3D positions
    coming from the simulator at the same time, UX takes priority so simulator is paused. This is implemented via
    a message Queue of limited buffer between the two processes. Putting messages is blocked until there's
    some available capacity.

    To run just create an instance and then call the method start() with your configuration.
    """

    def __init__(self):
        pass

    """
        Return a list of 3d points from the time series data generated after a step.

        Args:
            observer (CosimObserver): Cosim observer that will log the time-series data.
            signals (List): List of variables (signals) with metadata attached.
    """

    def get_step_values(
        self,
        observer: CosimObserver,
        signals: List[Tuple[Variable, Variable, Variable]],
    ) -> List[Tuple[float, float, float]]:
        variables = list(sum(signals, ()))
        slave_map: dict[int, List[int]] = {}

        for var in variables:
            if var.slave_index in slave_map:
                slave_map[var.slave_index].append(var.value_reference)
            else:
                slave_map[var.slave_index] = [var.value_reference]

        results: dict[str, float] = {}
        for slave_index, var_references in slave_map.items():
            time_series_values = observer.last_real_values(slave_index=slave_index, variable_references=var_references)

            for idx in range(len(time_series_values)):
                var_ref = var_references[idx]
                value = time_series_values[idx]
                results[f"{slave_index},{var_ref}"] = value

        cartesian_points: List[Tuple[float, float, float]] = []

        for x_signal, y_signal, z_signal in signals:
            x_pos = results[f"{x_signal.slave_index},{x_signal.value_reference}"]
            y_pos = results[f"{y_signal.slave_index},{y_signal.value_reference}"]
            z_pos = results[f"{z_signal.slave_index},{z_signal.value_reference}"]

            cartesian_points.append((x_pos, y_pos, z_pos))

        return cartesian_points

    """
        Add metadata to OSP signals such as the slave index and variable reference handled by the simulator.

        Args:
            instance (str): Simulator name as specified in the OSP System Structure file.
            port (str): Name of the signal from the model.
            slave_map (dict): Dictionary that maps models to slave indexes in the simulator.
            simulator (CosimExecution): Instance of the simulator
    """

    def variable_from_port(
        self,
        instance: str,
        port: str,
        slave_map: dict[Any, Any],
        simulator: CosimExecution,
    ) -> Variable:
        slave_index = slave_map[instance].index

        reference_dict = {
            var_ref.name.decode(): var_ref.reference for var_ref in simulator.slave_variables(slave_index)
        }
        var_reference = reference_dict[port]

        return Variable(
            name=port,
            instance=instance,
            value_reference=var_reference,
            slave_index=slave_index,
        )

    """
        Called by the simulation process upon initialization to create the simulation and step it. After each step time-series
        data is translated to 3D points following the specification defined by the points_3d parameter. Such data are then put
        to the message Queue

        Args:
            message_queue (Queue): Interprocess message queue instance
            points_3d (List[]): A list of tuples of size 3, each element in the tuple is another tuple that contains
                                the instance name (simulator name as specified in the OSP system structure file) and
                                the output port or signal. Threfore, a 3D point rendered is just the combination of
                                3 scalar signals.
            osp_xml (str): Path to OSP system structure file.
    """

    def run_simulation(
        self,
        message_queue: Queue,
        points_3d: List[Tuple[OSPSignal, OSPSignal, OSPSignal]],
        osp_xml: str = "",
    ):
        cosim_execution = CosimExecution.from_osp_config_file(osp_path=osp_xml)
        observer = CosimObserver.create_last_value()
        cosim_execution.add_observer(observer)
        cosim_execution.real_time_simulation_enabled()

        # Map model instance name to slave indexes
        slave_map = {slave_info.name.decode(): slave_info for slave_info in cosim_execution.slave_infos()}
        signals: List[tuple[Variable, Variable, Variable]] = []

        # Add metadata to 3d configuration signals and group them together following the same order
        for x_signal, y_signal, z_signal in points_3d:
            x_variable = self.variable_from_port(x_signal[0], x_signal[1], slave_map, cosim_execution)
            y_variable = self.variable_from_port(y_signal[0], y_signal[1], slave_map, cosim_execution)
            z_variable = self.variable_from_port(z_signal[0], z_signal[1], slave_map, cosim_execution)
            signals.append((x_variable, y_variable, z_variable))

        # Step the simulation and translate the time-series data before sharing it with the plotter via the message queue
        for currentTime in np.arange(0, 20, 0.1):  # 20
            cosim_execution.step()
            points = self.get_step_values(observer, signals)
            message_queue.put((currentTime, points), block=True)
            time.sleep(0.01)

    # Called by the plotter process to create the plot and refresh it when new data is shared via the message Queue
    def update_plot(self, queue: Queue):
        plt.ion()
        fig = plt.figure(figsize=(9, 9), layout="constrained")
        ax = Axes3D(fig=fig)
        # ax = plt.Axes(projection="3d")
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_zlim(0, 10)
        ax.view_init(elev=60, azim=45, roll=0)
        (line,) = ax.plot([0], [0], [0], linewidth=5)
        time_ax = ax.text(
            ax.get_xlim()[0],
            ax.get_ylim()[0],
            ax.get_zlim()[0],
            s="time=0",
            color="blue",
        )
        plot: Line3D = line

        while True:
            current_time, new_data = queue.get(block=True)
            if new_data is None:
                break
            new_data.insert(0, (0, 0, 0))
            x, y, z = zip(*new_data, strict=False)
            plot.set_data_3d(x, y, z)
            time_ax.set_text("time=" + str(round(current_time, 1)))
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

    """
        Kick off a visual simulation, triggering the creation of the two processes.

        Args:
            osp_system_structure (str): Path to the OSP system structure XML file
            points_3d (List[]): A list of tuples of size 3, each element in the tuple is another tuple that contains
                                the instance name (simulator name as specified in the OSP system structure file) and
                                the output port or signal. Threfore, a 3D point rendered is just the combination of
                                3 scalar signals.
    """

    def start(
        self,
        osp_system_structure: str,
        points_3d: List[Tuple[OSPSignal, OSPSignal, OSPSignal]],
    ):
        message_queue: Queue = Queue(maxsize=5)
        simulation_process = Process(
            target=self.run_simulation,
            args=(message_queue, points_3d, osp_system_structure),
        )
        plot_process = Process(target=self.update_plot, args=(message_queue,))

        simulation_process.start()
        plot_process.start()

        simulation_process.join()
        message_queue.put((None, None))  # Signal the plotter to stop
        plot_process.join()
