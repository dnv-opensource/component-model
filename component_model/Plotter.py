import numpy as np
import matplotlib.pyplot as plt
from component_model.model import Model
from component_model.variable import Variable_NP, spherical_to_cartesian

class Plotter(Model):
    def __init__(self, number_of_points: int = 1, **kwargs):
        super().__init__(name="Plotter",
                         description="Receives 3D signals and plots them",
                         author="Jorge Mendez",
                         version="0.1",
                         **kwargs)
        
        # Plot setup
        plt.ion()
        self.fig = plt.figure(figsize=(9,9), layout='constrained')
        ax = plt.axes(projection='3d')
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_zlim(0, 10)
        ax.view_init(elev=60, azim=45, roll=0)
        line, = ax.plot([0], [0], [0], linewidth=5)
        self.plot = line
        
        self.origin = Variable_NP(self, 'origin',
                                  'The starting anchor point of the plot in cartesian form, defaults to (0, 0, 0)',
                                  causality="parameter",
                                  initialVal=(0, 0, 0),
                                  rng=())
        
        
        self.points = [Variable_NP(self, f'point{i}',
                                    'The dimension and direction of the boom from anchor point to anchor point in m and spherical angles',
                                    causality='input',
                                    variability='continuous',
                                    initialVal=('0 m', '0deg', '0deg'),
                                    on_set=spherical_to_cartesian) for i in range(number_of_points)]
    
    def do_step(self, currentTime, stepSize):
        points = map(lambda point: point._value, self.points)
        absolutePoints = []

        for i, current in enumerate(points):
            if i == 0:
                absolutePoints.append(current)
            else:
                absolutePoints.append(absolutePoints[i-1] + current)

        x, y, z = zip(*absolutePoints)
        self.plot.set_data_3d(x, y, z)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        return True

class PlotterExample(Plotter):
    def __init__(self, **kwargs):
        super().__init__(number_of_points= 2, **kwargs)
