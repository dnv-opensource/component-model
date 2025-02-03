from math import sqrt
from typing import Any

import numpy as np
from pythonfmu import Fmi2Causality, Fmi2Slave, Real


class BouncingBall3D(Fmi2Slave):
    """Another Python-based BouncingBall model, using PythonFMU to construct a FMU.

    Features:

    * The ball has a 3-D vector as position and speed
    * As output variable the model estimates the next bouncing point
    * As input variables, the restitution coefficient `e`, the gravitational acceleration `g`
       and the initial speed can be changed.
    * Internal units are assumed as SI (m,s,rad)
    """

    def __init__(
        self,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        super().__init__(
            name="BouncingBall3D",
            description="Another Python-based BouncingBall model, using Model and Variable to construct a FMU",
            author="DNV, SEACo project",
            **kwargs,
        )

        # Register variable ports for the model
        self.posX = 0.0
        self.register_variable(Real("posX", causality=Fmi2Causality.output))

        self.posY = 0.0
        self.register_variable(Real("posY", causality=Fmi2Causality.output))

        self.posZ = 10.0
        self.register_variable(Real("posZ", causality=Fmi2Causality.output))

        self.speedX = 0.0
        self.register_variable(Real("speedX", causality=Fmi2Causality.output))

        self.speedY = 0.0
        self.register_variable(Real("speedY", causality=Fmi2Causality.output))

        self.speedZ = 0.0
        self.register_variable(Real("speedZ", causality=Fmi2Causality.output))

        self.p_bounceX = -1.0
        self.register_variable(Real("p_bounceX", causality=Fmi2Causality.output))

        self.p_bounceY = -1.0
        self.register_variable(Real("p_bounceY", causality=Fmi2Causality.output))
        self.t_bounce = -1.0

        self.g = 9.81  # Gravitational acceleration
        self.register_variable(Real("g", causality=Fmi2Causality.parameter))

        self.e = 0.0  # Coefficient of restitution
        self.register_variable(Real("e", causality=Fmi2Causality.parameter))

        # Internal states
        self.stopped = False
        self.min_speed_z = 1e-6
        self.accelerationX = 0.0
        self.accelerationY = 0.0
        self.accelerationZ = -self.g

    def do_step(self, _, dt):
        """Perform a simulation step from `self.time` to `self.time + dt`.

        With respect to bouncing (self.t_bounce should be initialized to a negative value)
        .t_bounce <= .time: update .t_bounce
        .time < .t_bounce <= .time+dt: bouncing happens within time step
        .t_bounce > .time+dt: no bouncing. Just advance .pos and .speed
        """
        if self.t_bounce < self.time:  # calculate first bounce
            self.t_bounce, self.p_bounce = self.next_bounce()
        while self.t_bounce <= self.time + dt:  # bounce happens within step or at border
            dt1 = self.t_bounce - self.time
            self.posX = self.p_bounceX
            self.posY = self.p_bounceY
            self.speedZ += self.accelerationZ * dt1  # speed before bouncing
            # speed reduction due to coefficient of restitution
            self.speedX *= self.e
            self.speedY *= self.e
            self.speedZ *= -self.e  # change also direction
            if self.speedZ < self.min_speed_z:
                self.stopped = True
                self.accelerationZ = 0.0
                self.speedZ = 0.0
                self.posZ = 0.0
            self.time += dt1  # jump to the exact bounce time
            dt -= dt1
            self.t_bounce, self.p_bounceX, self.p_bounceY = self.next_bounce()  # update to the next bounce
        if dt > 0:
            self.posX += self.speedX * dt
            self.posY += self.speedY * dt
            self.posZ += self.speedZ * dt + 0.5 * self.accelerationZ * dt**2
            self.speedZ += self.accelerationZ * dt
            self.time += dt
        self.posZ = max(self.posZ, 0)
        return True

    def next_bounce(self):
        """Calculate time of next bounce and position where the ground will be hit,
        based on .time, .pos and .speed.
        """
        if self.stopped:  # stopped bouncing
            return (1e300, np.array((1e300, 1e300, 0), float))
        dt_bounce = (self.speedZ + sqrt(self.speedZ**2 + 2 * self.g * self.posZ)) / self.g
        p_bounceX = self.posX + self.speedX * dt_bounce  # linear in x and y!
        p_bounceY = self.posY + self.speedY * dt_bounce
        return (self.time + dt_bounce, p_bounceX, p_bounceY)

    def setup_experiment(self, start: float):
        """Set initial (non-interface) variables."""
        super().setup_experiment(start)
        # print(f"SETUP_EXPERIMENT g={self.g}, e={self.e}")  # noqa: ERA001
        self.stopped = False
        self.time = start

    def exit_initialization_mode(self):
        """Initialize the model after initial variables are set."""
        super().exit_initialization_mode()
        self.accelerationZ = -self.g
