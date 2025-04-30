Introduction
============
The package extends the `PythonFMU package <https://github.com/NTNU-IHB/PythonFMU>`_.
It includes the necessary modules to construct a component model according to the fmi, OSP and DNV-RP-0513 standards
with focus on the following features:

* seamless translation of a Python model to an FMU package with minimal overhead (definition of FMU interface)
* support of vector variables (numpy)
* support of variable units and display units
* support of range checking of variables

Features which facilitate `Assurance of Simulation Models, DNV-RP-0513 <https://standards.dnv.com/explorer/document/6A4F5922251B496B9216572C23730D33/2>`_
shall have a special focus in this package.

Installation
------------

``pip install component-model``


Getting Started
---------------
A new model can consist of any python code. To turn the python code into an FMU the following is necessary

#. The model code is wrapped into a Python class which inherits from `Model`
#. The exposed interface variables (model parameters, input- and output connectors) are defined as `Variable` objects
#. The `(model).do_step( time, dt)` function of the model class is extended with model internal code,
   i.e. model evolves from `time` to `time+dt`.
#. Calling the method `Model.build()` will then compile the FMU and package it into a suitable FMU file.

See the files `example_models/bouncing_ball.py` and `tests/test_make_bouncingBall.py` supplied with this package
as a simple example of this process. The first file defines the model class
and the second file demonstrates the process of making the FMU and using it within fmpy and OSP.


Usage example
-------------
This is another BouncingBall example, using 3D vectors and units.

.. code-block:: python

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
         self.t_bounce, self.p_bounce = (-1.0, self.pos)  # provoke an update at simulation start

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



The following might be noted:

- The interface variables are defined in a separate local method ``_interface_variables``,
  keeping it separate from the model code.
- The ``do_step()`` method contains the essential code, describing how the ball moves through the air.
  It calls the ``super().do_step()`` method, which is essential to link it to ``Model``.
  The `return True` statement is also essential for the working of the emerging FMU.
- The ``next_bounce()`` method is a helper method.
- In addition to the extension of ``do_step()``, here also the ``setup_experiment()`` method is extended.
  Local (non-interface) variables can thus be initialized in a convenient way.

It should be self-evident that thorough testing of any model is necessary **before** translation to a FMU.
The simulation orchestration engine (e.g. OSP) used to run FMUs obfuscates error messages,
such that first stage assurance of a model should aways done using e.g. ``pytest``.
The minimal code to make the FMU file package is


.. code-block:: python

   from component_model.model import Model
   from fmpy.util import fmu_info

   asBuilt = Model.build("../component_model/example_models/bouncing_ball.py")
   info = fmu_info(asBuilt.name)  # not necessary, but it lists essential properties of the FMU


The model can then be run using `fmpy <https://pypi.org/project/FMPy/>`_

.. code-block:: python

   from fmpy import plot_result, simulate_fmu

   result = simulate_fmu(
      "BouncingBall.fmu",
      stop_time=3.0,
      step_size=0.1,
      validate=True,
      solver="Euler",
      debug_logging=True,
      logger=print,
      start_values={"pos[2]": 2}, # optional start value settings
   )
   plot_result(result)


Similarly, the model can be run using `OSP <https://opensimulationplatform.com/>`_
(or rather `libcosimpy <https://pypi.org/project/libcosimpy/>`_ - OSP wrapped into Python):

.. code-block:: Python

   from libcosimpy.CosimEnums import CosimExecutionState
   from libcosimpy.CosimExecution import CosimExecution
   from libcosimpy.CosimSlave import CosimLocalSlave

   sim = CosimExecution.from_step_size(step_size=1e7)  # empty execution object with fixed time step in nanos
   bb = CosimLocalSlave(fmu_path="./BouncingBall.fmu", instance_name="bb")

   print("SLAVE", bb, sim.status())

   ibb = sim.add_local_slave(bb)
   assert ibb == 0, f"local slave number {ibb}"

   reference_dict = {var_ref.name.decode(): var_ref.reference for var_ref in sim.slave_variables(ibb)}

   sim.real_initial_value(ibb, reference_dict["pos[2]"], 2.0)  # Set initial values

   sim_status = sim.status()
   assert sim_status.current_time == 0
   assert CosimExecutionState(sim_status.state) == CosimExecutionState.STOPPED
   infos = sim.slave_infos()
   print("INFOS", infos)

   sim.simulate_until(target_time=3e9)  # Simulate for 1 second


This is admittedly more complex than the ``fmpy`` example,
but it should be emphasised that fmpy is made for single component model simulation (testing),
while OSP is made for multi-component systems.


Development Setup
-----------------

1. Install uv
^^^^^^^^^^^^^
This project uses `uv` as package manager.

If you haven't already, install `uv <https://docs.astral.sh/uv/>`_, preferably using it's `"Standalone installer" <https://docs.astral.sh/uv/getting-started/installation/#__tabbed_1_2/>`_ method:

..on Windows:

``powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"``

..on MacOS and Linux:

``curl -LsSf https://astral.sh/uv/install.sh | sh``

(see `docs.astral.sh/uv <https://docs.astral.sh/uv/getting-started/installation//>`_ for all / alternative installation methods.)

Once installed, you can update `uv` to its latest version, anytime, by running:

``uv self update``

2. Install Python
^^^^^^^^^^^^^^^^^
This project requires Python 3.10 or later.

If you don't already have a compatible version installed on your machine, the probably most comfortable way to install Python is through ``uv``:

``uv python install``

This will install the latest stable version of Python into the uv Python directory, i.e. as a uv-managed version of Python.

Alternatively, and if you want a standalone version of Python on your machine, you can install Python either via ``winget``:

``winget install --id Python.Python``

or you can download and install Python from the `python.org <https://www.python.org/downloads//>`_ website.

3. Clone the repository
^^^^^^^^^^^^^^^^^^^^^^^
Clone the component-model repository into your local development directory:

``git clone https://github.com/dnv-opensource/component-model path/to/your/dev/component-model``

4. Install dependencies
^^^^^^^^^^^^^^^^^^^^^^^
Run ``uv sync`` to create a virtual environment and install all project dependencies into it:

``uv sync``

5. (Optional) Activate the virtual environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
When using ``uv``, there is in almost all cases no longer a need to manually activate the virtual environment.

``uv`` will find the ``.venv`` virtual environment in the working directory or any parent directory, and activate it on the fly whenever you run a command via `uv` inside your project folder structure:

``uv run <command>``

However, you still *can* manually activate the virtual environment if needed.
When developing in an IDE, for instance, this can in some cases be necessary depending on your IDE settings.
To manually activate the virtual environment, run one of the "known" legacy commands:

..on Windows:

``.venv\Scripts\activate.bat``

..on Linux:

``source .venv/bin/activate``

6. Install pre-commit hooks
^^^^^^^^^^^^^^^^^^^^^^^^^^^
The ``.pre-commit-config.yaml`` file in the project root directory contains a configuration for pre-commit hooks.
To install the pre-commit hooks defined therein in your local git repository, run:

``uv run pre-commit install``

All pre-commit hooks configured in ``.pre-commit-config.yam`` will now run each time you commit changes.

7. Test that the installation works
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To test that the installation works, run pytest in the project root folder:

``uv run pytest``


Meta
----
Copyright (c) 2025 `DNV <https://www.dnv.com/>`_ AS. All rights reserved.

Siegfried Eisinger - siegfried.eisinger@dnv.com

Distributed under the MIT license. See `LICENSE <LICENSE.md/>`_ for more information.

`https://github.com/dnv-opensource/component-model <https://github.com/dnv-opensource/component-model/>`_

Contribute
----------
Anybody in the FMU and OSP community is welcome to contribute to this code, to make it better,
and especially including other features from model assurance,
as we firmly believe that trust in our models is needed
if we want to base critical decisions on the support from these models.

To contribute, follow these steps:

1. Fork it `<https://github.com/dnv-opensource/component-model/fork/>`_
2. Create an issue in your GitHub repo
3. Create your branch based on the issue number and type (``git checkout -b issue-name``)
4. Evaluate and stage the changes you want to commit (``git add -i``)
5. Commit your changes (``git commit -am 'place a descriptive commit message here'``)
6. Push to the branch (``git push origin issue-name``)
7. Create a new Pull Request in GitHub

For your contribution, please make sure you follow the `STYLEGUIDE <STYLEGUIDE.md/>`_ before creating the Pull Request.
