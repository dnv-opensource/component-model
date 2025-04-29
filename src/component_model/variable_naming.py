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

    def __init__(self, varname: str, convention: VariableNamingConvention):
        self.parent: str | None  # None indicates no parent
        self.var: str
        self.indices: list[int] = []  # empty list indicates no indices
        self.der: int = 0  # 0 indicates 'no derivative'

        if convention == VariableNamingConvention.flat:  # expect python-conformant name (with indexing)
            var, indices = ParsedVariable.disect_indices(varname)
            self.parent = None
            self.var = var
            self.indices = indices
            self.der = 0
        else:  # structured variable naming (only these two are defined)
            m = re.match(r"der\((.+)\)", varname)
            if m is not None:
                vo = m.group(1)
                m = re.match(r"(.+),(\d+)$", vo)
                if m is not None:
                    var = m.group(1)
                    self.der = int(m.group(2))
                else:
                    var = vo
                    self.der = 1
            else:
                var = varname
                self.der = 0
            varlist = var.split(".")
            if len(varlist) > 1:
                self.parent = varlist[0] + "".join("." + varlist[i] for i in range(1, len(varlist) - 1))
                var = varlist[-1]
            else:
                self.parent = None

            self.var, self.indices = ParsedVariable.disect_indices(var)
        # assert self.var.isidentifier(), f"The variable name {self.var} is not a valid identifier"

    def as_tuple(self):
        """Return all fields as tuple."""
        return (self.parent, self.var, self.indices, self.der)

    @staticmethod
    def disect_indices(txt: str) -> tuple[str, list[int]]:
        m = re.match(r"(.+)\[([\d,]+)\]", txt)
        if m is None:
            return (txt, [])
        else:
            return (m.group(1), [int(t) for t in m.group(2).split(",")])
