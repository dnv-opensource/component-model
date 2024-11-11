from math import sqrt

import numpy as np

from component_model.model import Model
from component_model.variable import Variable


class BouncingBall3D(Model):
    """Another Python-based BouncingBall model, using PythonFMU to construct a FMU.

    Special features:

    * The ball has a 3-D vector as position and speed
    * As output variable the model estimates the next bouncing point
    * As input variables, the restitution coefficient `e`, the gravitational acceleration `g`
       and the initial speed can be changed.
    * Internal units are SI (m,s,rad)

    Args:
        pos (np.array)=(0,0,1): The 3-D position in of the ball at time [m]
        speed (np.array)=(1,0,0): The 3-D speed of the ball at time [m/s]
        g (float)=9.81: The gravitational acceleration [m/s^2]
        e (float)=0.9: The coefficient of restitution (dimensionless): |speed after| / |speed before| collision
        min_speed_z (float)=1e-6: The minimum speed in z-direction when bouncing stops [m/s]
    """

    def __init__(
        self,
        name: str = "BouncingBall3D",
        description="Another Python-based BouncingBall model, using Model and Variable to construct a FMU",
        pos: tuple = ("0 m", "0 m", "10 inch"),
        speed: tuple = ("1 m/s", "0 m/s", "0 m/s"),
        g: float = "9.81 m/s^2",
        e: float = 0.9,
        min_speed_z: float = 1e-6,
        **kwargs,
    ):
        super().__init__(name, description, author="DNV, SEACo project", **kwargs)
        self._pos = self._interface("pos", pos)
        self._speed = self._interface("speed", speed)
        self._g = self._interface("g", g)
        self.a = np.array((0, 0, -self.g), float)
        self._e = self._interface("e", e)
        self.min_speed_z = min_speed_z
        self.stopped = False
        self.time = 0.0
        self._p_bounce = self._interface("p_bounce", ("0m", "0m", "0m"))  # Note: 3D, but z always 0
        self.t_bounce, self.p_bounce = (
            -1.0,
            self.pos,
        )  # provoke an update at simulation start

    def do_step(self, _, dt):
        """Perform a simulation step from `self.time` to `self.time + dt`.

        With respect to bouncing (self.t_bounce should be initialized to a negative value)
        .t_bounce <= .time: update .t_bounce
        .time < .t_bounce <= .time+dt: bouncing happens within time step
        .t_bounce > .time+dt: no bouncing. Just advance .pos and .speed
        """
        if not super().do_step(self.time, dt):
            return False
        if self.t_bounce < self.time:  # calculate first bounce
            self.t_bounce, self.p_bounce = self.next_bounce()
        while self.t_bounce <= self.time + dt:  # bounce happens within step or at border
            dt1 = self.t_bounce - self.time
            self.pos = self.p_bounce
            self.speed += self.a * dt1  # speed before bouncing
            self.speed[2] = -self.speed[2]  # speed after bouncing if e==1.0
            self.speed *= self.e  # speed reduction due to coefficient of restitution
            if self.speed[2] < self.min_speed_z:
                self.stopped = True
                self.a[2] = 0.0
                self.speed[2] = 0.0
                self.pos[2] = 0.0
            self.time += dt1  # jump to the exact bounce time
            dt -= dt1
            self.t_bounce, self.p_bounce = self.next_bounce()  # update to the next bounce
        if dt > 0:
            # print(f"pos={self.pos}, speed={self.speed}, a={self.a}, dt={dt}")
            self.pos += self.speed * dt + 0.5 * self.a * dt**2
            self.speed += self.a * dt
            self.time += dt
        if self.pos[2] < 0:
            self.pos[2] = 0
        return True

    def next_bounce(self):
        """Calculate time of next bounce and position where the ground will be hit,
        based on .time, .pos and .speed.
        """
        if self.stopped:  # stopped bouncing
            return (1e300, np.array((1e300, 1e300, 0), float))
            # return ( float('inf'), np.array( (float('inf'), float('inf'), 0), float))
        else:
            dt_bounce = (self.speed[2] + sqrt(self.speed[2] ** 2 + 2 * self.g * self.pos[2])) / self.g
            p_bounce = self.pos + self.speed * dt_bounce  # linear. not correct for z-direction!
            p_bounce[2] = 0
            return (self.time + dt_bounce, p_bounce)

    def setup_experiment(self, start: float):
        """Set initial (non-interface) variables."""
        super().setup_experiment(start)
        self.stopped = False
        self.time = start

    def exit_initialization_mode(self):
        """Initialize the model after initial variables are set."""
        super().exit_initialization_mode()
        self.a = np.array((0, 0, -self.g), float)

    def _interface(self, name: str, start: float | tuple):
        """Define a FMU2 interface variable, using the variable interface.

        Args:
            name (str): base name of the variable
            start (str|float|tuple): start value of the variable (optionally with units)

        Returns:
            the variable object. As a side effect the variable value is made available as self.<name>
        """
        if name == "pos":
            return Variable(
                self,
                name="pos",
                description="The 3D position of the ball [m] (height in inch as displayUnit example.",
                causality="output",
                variability="continuous",
                initial="exact",
                start=start,
                rng=((0, "100 m"), None, (0, "10 m")),
            )
        elif name == "speed":
            return Variable(
                self,
                name="speed",
                description="The 3D speed of the ball, i.e. d pos / dt [m/s]",
                causality="output",
                variability="continuous",
                initial="exact",
                start=start,
                rng=((0, "1 m/s"), None, ("-100 m/s", "100 m/s")),
            )
        elif name == "g":
            return Variable(
                self,
                name="g",
                description="The gravitational acceleration (absolute value).",
                causality="parameter",
                variability="fixed",
                start=start,
                rng=(),
            )
        elif name == "e":
            return Variable(
                self,
                name="e",
                description="The coefficient of restitution, i.e. |speed after| / |speed before| bounce.",
                causality="parameter",
                variability="fixed",
                start=start,
                rng=(),
            )
        elif name == "p_bounce":
            return Variable(
                self,
                name="p_bounce",
                description="The expected position of the next bounce as 3D vector",
                causality="output",
                variability="continuous",
                start=start,
                rng=(),
            )
