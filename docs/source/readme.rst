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


1.	Install the `component_model` package: ``pip install component_model``
2.	Software dependencies: `PythonFMU`, `numpy`, `pint`, `uuid`, `ElementTree`
3.	Latest releases: Version 0.1, based on PythonFMU 0.64

Usage example
-------------
This is another BouncingBall example, using 3D vectors and units.

.. literalinclude:: ../../tests/examples/bouncing_ball_3d.py
   :language: python

The following might be noted:

* The interface variables are defined in a separate local method `_interface_variables`,
  keeping it separate from the model code.
* The `do_step()` method contains the essential code, describing how the ball moves through the air.
  It calls the `super().do_step()` method, which is essential to link it to `Model`.
  The `return True` statement is also essential for the working of the emerging FMU.
* The `next_bounce()` method is a helper method.
* In addition to the extension of `do_step()`, here also the `setup_experiment()` method is extended.
  Local (non-interface) variables can thus be initialized in a convenient way.

It should be self-evident that thorough testing of any model is necessary **before** translation to a FMU.
The simulation orchestration engine (e.g. OSP) used to run FMUs obfuscates error messages, 
such that first stage assurance of a model should aways done using e.g. `pytest`.

The minimal code to make the FMU file package is 

.. code-block:: Python

   from component_model.model import Model
   from fmpy.util import fmu_info

   asBuilt = Model.build("../component_model/example_models/bouncing_ball.py")
   info = fmu_info(asBuilt.name)  # not necessary, but it lists essential properties of the FMU

The model can then be run using `fmpy <https://pypi.org/project/FMPy/>`_ 

.. code-block:: Python

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

   # Set initial values
   sim.real_initial_value(ibb, reference_dict["pos[2]"], 2.0)

   sim_status = sim.status()
   assert sim_status.current_time == 0
   assert CosimExecutionState(sim_status.state) == CosimExecutionState.STOPPED
   infos = sim.slave_infos()
   print("INFOS", infos)

   # Simulate for 1 second
   sim.simulate_until(target_time=3e9)

This is admittedly more complex than the `fmpy` example,
but it should be emphasised that fmpy is made for single component model simulation (testing),
while OSP is made for multi-component systems.

Contribute
----------
Anybody in the FMU and OSP community is welcome to contribute to this code, to make it better, 
and especially including other features from model assurance, 
as we firmly believe that trust in our models is needed 
if we want to base critical decisions on the support from these models.