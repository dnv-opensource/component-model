import time
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from libcosimpy.CosimExecution import CosimExecution
from libcosimpy.CosimObserver import CosimObserver
from typing import List, Tuple, Any
from dataclasses import dataclass
from libcosimpy.CosimEnums import CosimVariableCausality, CosimVariableType
import matplotlib.animation as animation
from multiprocessing import Process, Queue
import os
import numpy as np

from mpl_toolkits.mplot3d.art3d import Line3D
from enum import Enum

class SimulatorStatus(Enum):
    stopped = 0
    stepping = 1
    completed = 2

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

OSPSignal = Tuple[str, str]

class VisualSimulator:
    def __init__(self):
        pass

    def get_step_values(self, observer: CosimObserver, signals:  List[Tuple[Variable, Variable, Variable]]) -> List[Tuple[float, float, float]]:
        variables = list(sum(signals, ()))
        slave_map : dict[int, List[int]] = {}

        for var in variables:
            if var.slave_index in slave_map:
                slave_map[var.slave_index].append(var.value_reference)
            else:
                slave_map[var.slave_index] = [var.value_reference]

        results: dict[str,float] = {}
        for slave_index, var_references in slave_map.items():
            time_series_values = observer.last_real_values(slave_index=slave_index, variable_references=var_references)
            
            for idx in range(len(time_series_values)):
                var_ref = var_references[idx]
                value = time_series_values[idx]
                results[f'{slave_index},{var_ref}'] = value

        cartesian_points: List[Tuple[float, float, float]] = []

        for x_signal, y_signal, z_signal in signals:
            x_pos = results[f'{x_signal.slave_index},{x_signal.value_reference}']
            y_pos = results[f'{y_signal.slave_index},{y_signal.value_reference}']
            z_pos = results[f'{z_signal.slave_index},{z_signal.value_reference}']

            cartesian_points.append((x_pos, y_pos, z_pos))

        return cartesian_points
    
    def variable_from_port(self, instance: str, port: str, slave_map: dict[Any, Any], simulator: CosimExecution) -> Variable:
        slave_index = slave_map[instance].index

        reference_dict = {var_ref.name.decode(): var_ref.reference for var_ref in simulator.slave_variables(slave_index)}
        var_reference = reference_dict[port]

        return Variable(
            name= port,
            instance= instance,
            value_reference= var_reference,
            slave_index= slave_index,
        )

    def run_simulation(self, message_queue: Queue, points_3d: List[Tuple[OSPSignal, OSPSignal, OSPSignal]], osp_xml: str = ''):
        cosim_execution = CosimExecution.from_osp_config_file(osp_path=osp_xml)
        observer = CosimObserver.create_last_value()
        cosim_execution.add_observer(observer)
        cosim_execution.real_time_simulation_enabled()

        slave_map = {slave_info.name.decode(): slave_info for slave_info in cosim_execution.slave_infos()}
        signals = []
        for x_signal, y_signal, z_signal in points_3d:
            x_variable = self.variable_from_port(x_signal[0], x_signal[1], slave_map, cosim_execution)
            y_variable = self.variable_from_port(y_signal[0], y_signal[1], slave_map, cosim_execution)
            z_variable = self.variable_from_port(z_signal[0], z_signal[1], slave_map, cosim_execution)
            signals.append((x_variable, y_variable, z_variable))

        for currentTime in np.arange(0, 20, 0.1): #20
            cosim_execution.step()
            points = self.get_step_values(observer, signals)
            message_queue.put((currentTime, points), block=True)
            time.sleep(0.01)

    def update_plot(self, queue: Queue):
        plt.ion()
        fig = plt.figure(figsize=(9,9), layout='constrained')
        ax = plt.axes(projection='3d')
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_zlim(0, 10)
        ax.view_init(elev=60, azim=45, roll=0)
        line, = ax.plot([0], [0], [0], linewidth=5)
        time_ax = ax.text( ax.get_xlim()[0], ax.get_ylim()[0], ax.get_zlim()[0], s='time=0', color='blue')
        plot = line

        while True:
            curren_time, new_data = queue.get(block=True)
            if new_data is None:
                break
            new_data.insert(0, (0, 0, 0))
            x, y, z = zip(*new_data)
            plot.set_data_3d(x, y, z)
            time_ax.set_text('time='+str(round(curren_time, 1)))
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

    def start(self, osp_system_structure: str, points_3d: List[Tuple[OSPSignal, OSPSignal, OSPSignal]]):
        message_queue = Queue(maxsize=5)
        simulation_process = Process(target=self.run_simulation, args=(message_queue, points_3d, osp_system_structure))
        plot_process = Process(target=self.update_plot, args=(message_queue,))

        simulation_process.start()
        plot_process.start()

        simulation_process.join()
        message_queue.put((None, None))  # Signal the plotter to stop
        plot_process.join()

if __name__ == '__main__':
    ss_path = os.path.join( os.path.abspath( os.path.curdir), 'OspSystemStructure.xml')
    plotter = VisualSimulator()
    plotter.start(osp_system_structure=ss_path, points_3d=[(('mobileCrane', 'pedestal.cartesianEnd.0'), ('mobileCrane', 'pedestal.cartesianEnd.1'), ('mobileCrane', 'pedestal.cartesianEnd.2')),
                                                               (('mobileCrane', 'rope.cartesianEnd.0'), ('mobileCrane', 'rope.cartesianEnd.1'), ('mobileCrane', 'rope.cartesianEnd.2'))])
