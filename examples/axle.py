import logging
from math import cos, pi, sin

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Axle:
    """Car with two wheels connected by fixed rod of length a.

    Args:
        d1 (float)= 1.0: Diameter of wheel1 of the axle
        rpm1 (float)= 1.0: Angle-speed of wheel2 (revolvements per second) of wheel1
        d2 (float)= 1.0: Diameter of wheel2 of the axle
        rpm2 (float)= 1.0: Angle-speed (revolvements per second) of wheel2
        a (float)= 1.0: axle length of the axle
    """

    def __init__(self, d1: float = 1.0, rpm1: float = -1.0, d2: float = 1.0, rpm2: float = -1.0, a: float = 1.0):
        self.wheels = [Wheel(d1, rpm1, pos=np.array([0.0, 0.0], float)), Wheel(d2, rpm2, pos=np.array([a, 0.0], float))]
        self.a = a
        self.time = 0.0
        self.times: list

    def init_drive(self):
        self.wheels[0].pos = np.array([0.0, 0.0], float)
        self.wheels[1].pos = np.array([self.a, 0.0], float)
        for i in range(2):
            self.wheels[i].track = None
        self.times = []

    def drive(self, time: float, dt: float):
        self.times.append(time)
        pos = [self.wheels[i].pos for i in range(2)]
        b = [self.wheels[i].length(dt) for i in range(2)]
        axle = pos[1] - pos[0]
        if abs(b[0] - b[1]) > 1e-10:  # on circle
            if abs(b[0]) < 1e-10:
                r = 0.0
                center = pos[0]
                angle = b[1] / self.a
            elif abs(b[1]) < 1e-10:
                r = -self.a
                center = pos[1]
                angle = -b[0] / self.a
            else:
                r = 1.0 / (1.0 - b[1] / b[0])  # in units of self.a
                center = pos[0] + r * axle
                angle = b[0] / (r * self.a)
            sa = sin(angle)
            ca = cos(angle)
            rot = np.array([[ca, -sa], [sa, ca]], float)
            for i in range(2):
                pos[i] = np.matmul(rot, (pos[i] - center)) + center
        else:  # nearly equal. Radius very large. Straight ahead
            direction = np.array([axle[1], -axle[0]], float) / self.a
            for i in range(2):
                pos[i] += b[i] * direction
        for i in range(2):
            self.wheels[i].pos = pos[i]
        self.time += dt

    def show(self):
        fig, ax = plt.subplots()
        (x1, y1) = self.wheels[0].track
        ax.plot(x1, y1, label="1")
        (x2, y2) = self.wheels[1].track
        ax.plot(x2, y2, label="2")
        ax.legend()
        plt.show()


class Wheel:
    """Wheel with diameter and motor."""

    def __init__(self, diameter: float = 1.0, rpm: float = -1.0, pos: np.ndarray | None = None):
        self.diameter = diameter
        self.motor = Motor(rpm)
        self._pos = np.array([0, 0], float) if pos is None else pos
        self._track: list[list[float]] = [[], []]

    def length(self, dt: float):
        return pi * self.diameter * self.motor.angle(dt)

    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, newpos: list[float] | np.ndarray):
        self.track = newpos
        self._pos = np.array(newpos, float)

    @property
    def track(self):
        return self._track

    @track.setter
    def track(self, newpos: list[float] | np.ndarray | None):
        if newpos is None:  # reset
            self._track = [[], []]
        else:
            for i in range(2):
                self._track[i].append(newpos[i])


class Motor:
    """Motor turning with rpm angular speed (revolvements per second using right hand rule)."""

    def __init__(self, rpm: float = -1.0):
        self.rpm = rpm  # current angular speed
        self.acc = 0.0  # current angular acceleration

    def angle(self, dt: float):
        """Calculate the total angle the motor turns."""
        if self.acc == 0.0:
            return self.rpm * dt
        else:
            rpm0 = self.rpm
            self.rpm += self.acc * dt  # new rpm after the step
            return rpm0 * dt + 0.5 * self.acc * dt * dt
