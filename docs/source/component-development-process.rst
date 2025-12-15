*******************************************************
Guideline on how to develop a FMU using component-model
*******************************************************

The development process follows the steps

#. Develop a functional model using Python (>=3.10.) as a Python class. Called *basic model* here.
#. Thoroughly test the basic model. 
#. Define the FMU interface using component-model functions
#. Build the FMU calling `Model.build()`, optionally overwriting optional argument of the model class.
#. Test the FMU standalone, e.g. using the `FMPy` package or in conjunction with other FMUs using the `sim-explorer` package.

Develop a basic model
=====================
Define a functional model as a Python class. We refer to this model as the *basic model*.
At this stage the emerging model class does not need to refer to the `component-model` package. 
In fact it is **not recommended** to derive the basic model from `Model`class of component-model. 
The basic model might import any Python package (e.g. numpy), as needed to satisfy the functionality.

Testing the basic model
=======================
The basic model should be thoroughly tested. 
This cannot be emphasised too much, as test possibilities and feadback is limited in the FMU domain,
while Python offers proper test and debugging facilities.


Defining the FMU interface
==========================
A FMU interface must be added to the model prior to package the model as FMU. This concerns basically

* component model parameters, i.e. settings which can be changed prior to a simulation run, but are typically constant during the run
* component model input variables, i.e. variables which can be changed based on the output from other component models
* component model output variables, i.e. results which are provided to any connected component model, or the system.

Defining the interface is done like

.. code-block:: Python

    class <basic-model>_FMU(Model, <basic-model>):
        def __init__(self, <basic_model_args, **kwargs):
            Model.__init__(self,name,description,author,version, kwargs)
            <basic-model>.__init__(<basic-model-args>)


Virtual derivatives
-------------------
Running component models in scenarios it is often necessary to change variables during the simulation run.
As in the reality it is often not a good idea to step values in huge steps, as this resembles a 'hammer stroke',
which the system might not tolerate. The simulation dynamics does often not handle such a situation properly either.
It is therefore often necessary to ramp up or down values to the desired final values, 
i.e. changing the derivative of the input variable to a non-zero value
until the desired value of the parent variable is reached and then setting the derivative back to zero.

It is cumbersome to introduce derivative variables for every parent variable which might be changed during the simulation.
Therefore. component-model introduces the concept of *virtual derivatives*, 
which are derivative interface variables which are not linked to variables in the basic model.
When defining such variables and setting them to non-zero values, 
the parent variable is changed with the given slope at every time step,
i.e. 

`<parent-variable> += d<parent-variable>/dt * <step-size>`

where `d<parent-variable>/dt` is the non-zero derivative value (slope).

In practical terms, virtual derivatives are defined using FMI structured variables syntax:

.. code-block:: Python

    Variable( name='der(<parent-variable-name>)', causality='input', variability='continuous', ...)

Explicit specification of the arguments `owner` and `local_name` should be avoided.
Specification of `local_name` changes the virtual derivative into a non-virtual derivative,
i.e. the variable is expected to exist in the basic model.
The `on_step` argument is automatically set to change the parent variable at every time step if the derivative is non-zero.
Explicitly overwriting the automatic `on_step` function is allowed at one's own expense.


Building the FMU
================

Testing the FMU
===============