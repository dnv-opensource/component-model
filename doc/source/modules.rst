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
Python module to define a model interface. It extends the PythonFMU.variables class.

It is important to understand the way that variables are defined and accessed:

* From the outside (the system FMI view), interface variables are accessed through `fmi2Getxxx` and `fmi2Setxxx`, 
  where `xxx` denotes the variable type (like `Real`). 
  The `Get` and `Set` functions use valueReferences (lists of integers) to access variables.
* Variables have also a name, unique within the component model. Within the `component_model` package we establish the convention:

   #. Interface `Variable` objects are members of the `Model` object accessible through their name. 
      E.g. `crane.pedestal_mass` denotes the mass variable of the crane pedestal.
   #. Compound variable objects like `VariableNP` are accessible through their base name, i.e. omitting the index `[i].
      E.g. `crane.pedestal_end` denotes the 3-d cartesian end point of the crane pedestal, 
      while e.g. `pedestal_end[0]` is listed as the `ScalarVariable` x component in `modelDescription.xml` with its own `valueReference`.
   #. The value of variables is managed by the `Value` class...

   #. The `start` member of the `Variable` object is used both to provide a default initial value of the variable 
      and (optionally) to provide units to the variable. `value0` may thus be provided as string, while the variable is e.g. of float type.
      The units of compound variable components do not need to be equal. See component_model.variableNP_ for details.
   #. The `_val(val)` member of `Variable` is used to access (@property val) and set (@setter.val) the value of the variable.
      This function is intended for use internally in the model. No unit conversion, range checking and `on set` actions are performed.
   #. The `_value(val)` member of `Variable`is similar to `_val`, but is intended for use from outside the model,
      i.e. this is the function run by `fmi2Getxxx` and `fmi2Setxxx` (see above), unit conversion is performed, ranges are checked 
      and `on set` actions are run.
   #. Optionally an `on set(val)` hook can be defined as action to be performed each time the variable is changed through `_value()`.
      For scalar variables this is rather superfluous, since such actions can be included in `_value()`, but for compound variables...


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

   