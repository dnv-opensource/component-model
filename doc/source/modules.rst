Modules documentation
=====================
This section documents the contents of the Component Model package.


Model
-----
Python module as container for a (FMU) component model. 

.. autoclass:: component_model.model
   :members:

Variable
--------
Python module to define a model interface. It extends the PythonFMU.variables class

.. autoclass:: component_model.variable
   :members:

.. autoclass:: component_model.variableNP
   :members:

Wrap_FMU
---------
Python module to wrap a component model into a PythonFMU FMU model.
TODO: Not sure whether this should be a separate class, or whether it should a function within Model.

.. autoclass:: component_model.wrapFMU
   :members:

   