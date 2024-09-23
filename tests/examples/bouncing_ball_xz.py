from math import sqrt

import numpy as np
from component_model.model import Model
from component_model.variable import Variable


class BouncingBallXZ(Model):
    """Fmi2Slave implementation of a model made in Python, performing the FMU 'packaging', implements the pythonfmu.Fmi2Slave and runs buildFMU, i.e.
          * prepare the modeldescription.xml
          * implement the FMI2 C interface for the present platform (as .dll, .so, ...).

       The following is expected of any valid Python model:
          * a complete list of variables including meta information, the model.variables dictionary contains that
          * a do_step method specifying what happens at each simulation step

    Args:
       model (obj): reference to the (instantiated) model object

       licenseTxt (str)='Open Source'
       copyright (str)='See copyright notice of used tools'
       defaultExperiment (dict) = None: key/value dictionary for the default experiment setup
       guid (str)=None: Unique identifier of the model (supplied or automatically generated)
       non_default_flags (dict)={}: Any of the defined FMI flags with a non-default value (see FMI 2.0.4, Section 4.3.1)
       **kwargs: Any other keyword argument (transferred to the basic slave object)

    """

    def __init__(
        self,
        name="BouncingBallXZ",
        description="Simple bouncing ball test FMU",
        author="DNV, SEACo project",
        version="0.1",
        defaultExperiment: dict | None = None,
        guid="06128d688f4d404d8f6d49d6e493946b",
        v_min=1e-15,
        **kwargs,
    ):
        if defaultExperiment is None:
            defaultExperiment = {"start_time": 0.0, "step_size": 0.1, "stop_time": 10.0, "tolerance": 0.001}
        super().__init__(
            name=name,
            description=description,
            author=author,
            version=version,
            defaultExperiment=defaultExperiment,
            **kwargs,
        )
        self.v_min = v_min
        Variable(
            self,
            start=(0.0, 0.0),
            name="x",
            description="""Position of ball (x,z) at time.""",
            causality="output",
            variability="continuous",
            initial="exact",
        )
        Variable(
            self,
            start=(1.0, 1.0),
            name="v",
            description="speed at time as (x,z) vector",
            causality="output",
            variability="continuous",
            initial="exact",
        )
        Variable(
            self,
            start=(1.0, 1.0),
            name="v0",
            description="speed at time=0 as (x,z) vector",
            causality="parameter",
            variability="fixed",
            initial="exact",
        )
        Variable(
            self,
            start=0.95,
            name="bounceFactor",
            description="factor on speed when bouncing",
            causality="parameter",
            variability="fixed",
        )
        Variable(
            self,
            start=0.0,
            name="drag",
            description="drag decelleration factor defined as a = self.drag* v^2 with dimension 1/m",
            causality="parameter",
            variability="fixed",
        )
        Variable(
            self,
            start=0.0,
            name="energy",
            description="Total energy of ball in J",
            causality="output",
            variability="continuous",
        )
        Variable(
            self,
            start=0.0,
            name="period",
            description="Bouncing period of ball",
            causality="output",
            variability="continuous",
        )

    #        self.register_variable( String("mdShort", causality=Fmi2Causality.local))

    def enter_initialization_mode(self):
        self.v = self.v0
        self.energy = 9.81 * self.x[1] + 0.5 * np.dot(self.v, self.v)
        self.period = 2 * self.v[1] / 9.81  # may change when energy is taken out of the system
        return True

    def do_step(self, current_time, step_size):
        #        print("ENERGY", self.energy, type(self.energy))
        def bounce_loss(v0):
            if self.bounceFactor == 1.0:
                return v0
            if abs(v0) < self.v_min:
                v0 = 0.0
                self.energy = 0.0
                self.period = 0.0
            else:
                v0 *= self.bounceFactor  # speed with which it leaves the ground
                self.energy = v0 * v0 / 2
                self.period = 2 * v0 / 9.81
            return v0

        self.x[0] += self.v[0] * step_size
        y = self.x[1] + self.v[1] * step_size - 9.81 / 2 * step_size**2
        if y <= 0 and self.v[1] < 0:  # bounce
            # time when it hits the ground:
            t0 = self.v[1] / 9.81 * (1 - sqrt(1 + 2 * self.x[1] * 9.81 / self.v[1] ** 2))
            # more exact than self.v_y - 9.81* t0 # speed when jumps off the ground (without energy loss):
            v0 = sqrt(2 * self.energy)
            v0 = bounce_loss(v0)  # check energy loss during bouncing
            # print("BOUNCE", current_time, self.x.value , t0, v0)
            tRest = step_size - t0
            while True:
                if tRest < self.period:  # cannot do a whole bounce in the remaining time
                    break
                if self.drag != 0:
                    raise NotImplementedError(
                        "Bouncing a whole period is not implemented when drag is involved. Try choosing smaller time steps."
                    )
                v0 = bounce_loss(v0)
                if v0 == 0.0:  # movement stopped (numerical issues would otherwise start an oscillation)
                    tRest = 0.0
                    break
                else:
                    tRest -= self.period

            self.x[1] = v0 * tRest - 9.81 / 2 * tRest * tRest  # height end of step
            self.v[1] = v0 - 9.81 * tRest  # speed end of step
        else:
            self.v[1] -= 9.81 * step_size
            self.x[1] = y
        if self.drag != 0:
            fac = 1 - self.drag * np.linalg.norm(self.v) * step_size
            self.v *= fac
            self.energy = 9.81 * self.x[0] + 0.5 * np.dot(self.v, self.v)
            # print("FAC", fac, self.v, self.energy)
        #         e = 9.81*self.x[1] + 0.5*self.v[1]**2
        #         if  abs( e-self.energy) > 1e-6: # and  and
        #             print("Energy leak", current_time, e, self.energy)
        #             self.energy = e
        return True
