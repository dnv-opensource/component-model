import xml.etree.ElementTree as ET  # noqa: N817
from pathlib import Path


# ==========================================
# Open Simulation Platform related functions
# ==========================================
def make_osp_system_structure(
    name: str = "OspSystemStructure",
    models: dict | None = None,
    connections: tuple = (),
    version: str = "0.1",
    start: float = 0.0,
    base_step: float = 0.01,
    algorithm: str = "fixedStep",
    path: Path | str = ".",
):
    """Prepare a OspSystemStructure xml file according to `OSP configuration specification <https://open-simulation-platform.github.io/libcosim/configuration>`_.

    Args:
        name (str)='OspSystemStructure': the name of the system model, used also as file name
        models (dict)={}: dict of models (in OSP called 'simulators').
          A model is represented by a dict element modelName : {property:prop, variable:value, ...}
        connections (tuple)=(): tuple of model connections.
          Each connection is defined through a tuple of (model, variable, model, variable),
          where variable can be a tuple defining a variable group
        version (str)='0.1': The version of the system model
        start (float)=0.0: The simulation start time
        base_step (float)=0.01: The base stepSize of the simulation. The exact usage depends on the algorithm chosen
        algorithm (str)='fixedStep': The name of the algorithm
        path (Path,str)='.': the path where the file should be saved

    Returns
    -------
        The absolute path of the file as Path object

        .. todo:: better stepSize control in dependence on algorithm selected, e.g. with fixedStep we should probably set all step sizes to the minimum of everything?
    """

    def make_simulators():
        """Make the <simulators> element (list of component models)."""

        def make_initial_value(var: str, val: bool | int | float | str):
            """Make a <InitialValue> element from the provided var dict."""
            _type = {bool: "Boolean", int: "Integer", float: "Real", str: "String"}[type(val)]
            initial = ET.Element("InitialValue", {"variable": var})
            ET.SubElement(
                initial,
                _type,
                {"value": ("true" if val else "false") if isinstance(val, bool) else str(val)},
            )
            return initial

        simulators = ET.Element("Simulators")
        if len(models):
            for m, props in models.items():
                # Note: instantiated model names might be small, but FMUs are based on class names and are therefore capitalized
                simulator = ET.Element(
                    "Simulator",
                    {
                        "name": m,
                        "source": props.get("source", m[0].upper() + m[1:] + ".fmu"),
                        "stepSize": str(props.get("stepSize", base_step)),
                    },
                )
                initial = ET.SubElement(simulator, "InitialValues")
                for prop, value in props.items():
                    if prop not in ("source", "stepSize"):
                        initial.append(make_initial_value(prop, value))
                simulators.append(simulator)
            #            print(f"Model {m}: {simulator}. Length {len(simulators)}")
            #            ET.ElementTree(simulators).write("Test.xml")
            return simulators

    def make_connections():
        """Make the <connections> element from the provided con."""
        cons = ET.Element("Connections")
        if len(connections):
            m1, v1, m2, v2 = connections
            if isinstance(v1, (tuple, list)):  # group connection (e.g. a compound Variable)
                if not isinstance(v2, (tuple, list)) or len(v2) != len(v1):
                    raise Exception(f"Vector connection between {m1} and {m2} does not match.") from None
                for i in range(len(v1)):
                    con = ET.Element("VariableConnection")
                    ET.SubElement(con, "Variable", {"simulator": m1, "name": v1[i]})
                    ET.SubElement(con, "Variable", {"simulator": m2, "name": v2[i]})
                    cons.append(con)
            else:  # single connection
                con = ET.Element("VariableConnection")
                ET.SubElement(con, "Variable", {"simulator": m1, "name": v1})
                ET.SubElement(con, "Variable", {"simulator": m2, "name": v2})
                cons.append(con)
        return cons

    osp = ET.Element(
        "OspSystemStructure",
        {
            "xmlns": "http://opensimulationplatform.com/MSMI/OSPSystemStructure",
            "version": version,
        },
    )
    osp.append(make_simulators())
    osp.append(make_connections())
    tree = ET.ElementTree(osp)
    ET.indent(tree, space="   ", level=0)
    file = Path(path).absolute() / (name + ".xml")
    tree.write(file, encoding="utf-8")
    return file
