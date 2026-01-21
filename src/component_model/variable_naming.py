"""Parse variable names using variableNamingConvention 'flat' or 'structured'
as defined in FMI2.0, 2.2.9 and FMI3.0, 2.4.7.5.1. The definition has not changed.
"""

import re
from enum import Enum


class VariableNamingConvention(Enum):
    """Enum for variable naming conventions."""

    flat = 0
    structured = 1


class ParsedVariable:
    """Parse the varname with respect to the given VariableNamingConvention.

    Note: The parent hierarchy (if present) is expected to refer to the objects owning the various variables,
      It does not refer to variable names, as the FMI examples suggest.
    Results:
        dict containing the keys parent (full

        * parent: full parent name or None,
        * var: basic variable name,
        * indices: list of indices (int) as defined in FMI standard or empty list,
        * der: unsigned integer, defining the derivation order. 0 for no derivation
    """

    def __init__(self, varname: str, convention: Enum | None = VariableNamingConvention.structured):
        self.parent: str | None  # None indicates no parent
        self.var: str
        self.indices: list[int] = []  # empty list indicates no indices
        self.der: int = 0  # 0 indicates 'no derivative'

        if (
            convention is None or convention == VariableNamingConvention.flat
        ):  # expect (indexed) python-conformant names
            var, indices = ParsedVariable.disect_indices(varname)
            self.parent = None
            self.var = var
            self.indices = indices
            self.der = 0
        elif convention == VariableNamingConvention.structured:  # structured variable naming
            self.der = 0  # default and count start
            var = varname
            while True:
                m = re.match(r"der\((.+)\)", var)  # der(*)
                if m is None:
                    break
                vo = m.group(1)
                m = re.match(r"(.+),\s*(\d+)$", vo)  # check order of der
                if m is not None:
                    var = m.group(1)
                    self.der += int(m.group(2))
                else:
                    var = vo
                    self.der += 1
            varlist = var.split(".")
            if len(varlist) > 1:
                self.parent = varlist[0] + "".join("." + varlist[i] for i in range(1, len(varlist) - 1))
                var = varlist[-1]
            else:
                self.parent = None

            self.var, self.indices = ParsedVariable.disect_indices(var)
        else:
            raise ValueError("VariableNamingConvention Enum expected. Got {convention}")

    def as_tuple(self):
        """Return all fields as tuple."""
        return (self.parent, self.var, self.indices, self.der)

    def as_string(
        self,
        include: tuple[str, ...] = ("parent", "var", "indices", "der"),
        simplified: bool = True,
        primitive: bool = False,
        index: str = "",
    ):
        """Re-construct the variable name, including what is requested and optionally adapting.

        Args:
            include (tuple): list of strings of what to incdule of "parent", "var", "indices", "der"
            simplified (bool)= True: Optionally change the style to also including superfluous info
            primitive (bool)= False: If true, 'der' included and >0, determines the name of the primitive.
            index (str)="": If !="" and 'indices' included, specifies the index to use

        This is convenient to e.g. leave out indices (vector variables) or finding parent names of derivatives.
        """
        if "parent" in include:
            name = "" if self.parent is None else self.parent
        else:
            name = ""
        if "var" in include:
            name = name + "." + self.var if len(name) else self.var
        if "indices" in include and (len(self.indices) or index != ""):  # never empty parantheses
            if index != "":
                name += f"[{index}]"
            else:
                name += str(self.indices)
        if "der" in include and self.der > 0:
            der = self.der if not primitive else self.der - 1
            if der == 0:
                pass  # just the name
            elif not simplified or der > 1:
                name = f"der({name},{der})"
            elif simplified and der == 1:
                name = f"der({name})"
            else:
                raise NotImplementedError(f"Unknown combination simplified={simplified}, der={der}") from None
        return name

    @staticmethod
    def disect_indices(txt: str) -> tuple[str, list[int]]:
        m = re.match(r"(.+)\[([\d,\s*]+)\]", txt)
        if m is None:
            return (txt, [])
        else:
            return (m.group(1), [int(t) for t in m.group(2).split(",")])
