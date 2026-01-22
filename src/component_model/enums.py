"""Additional Enum objects for component-model and enum-related utilities."""

import logging
from enum import Enum, EnumType, IntFlag

from pythonfmu.enums import Fmi2Causality as Causality
from pythonfmu.enums import Fmi2Initial as Initial
from pythonfmu.enums import Fmi2Variability as Variability

logger = logging.getLogger(__name__)


def ensure_enum(org: str | Enum | None, default: Enum | EnumType | None) -> Enum | None:
    """Ensure that we have an Enum, based on the input as str, Enum or None."""
    if org is None and default is None:
        return None
        raise ValueError("org and default shall not both be None") from None
    elif org is None:
        assert isinstance(default, Enum), f"Need an Enum (member) as default if org=None. Found {type(default)}"
        return default
    elif default is None:
        assert isinstance(org, Enum), "When no default is provided, org must be an Enum."
        return org
    elif isinstance(org, str):  # both provided and org is a string
        _default = default if isinstance(default, EnumType) else type(default)  # need the Enum itself
        if org in _default.__members__:
            e: Enum = _default[org]  # type: ignore[reportAssignmentType]
            assert isinstance(e, Enum)
            return e
        else:
            raise Exception(f"The value {org} is not compatible with the Enum {_default}") from None
    else:  # expect already an EnumType
        assert isinstance(org, type(default)), f"{org} is not member of the Enum {type(default)}"
        return org


# see tables on page 50 in FMI 2.0.1 specification
combinations = (
    # causality:
    # parameter, c.par, input, output, local, independent
    #                                    Variability
    ("a", "a", "a", "A", "A", "c"),  # constant
    ("A", "B", "d", "e", "B", "c"),  # fixed
    ("A", "B", "d", "e", "B", "c"),  # tunable
    ("b", "b", "D", "C", "C", "c"),  # discrete
    ("b", "b", "D", "C", "C", "E"),  # continuous
)


def combination(var: Enum, caus: Enum):
    return combinations[var.value][caus.value]


def use_start(causality: Causality | None, variability: Variability | None, initial: Initial | None) -> bool:
    if causality is None or variability is None:
        raise ValueError(f"None of the parameters {causality}, {variability} should be None") from None
    use = (
        initial in (Initial.exact, Initial.approx)
        or causality in (Causality.parameter, Causality.input)
        or variability == Variability.constant
        or (causality in (Causality.output, Causality.local) and initial != Initial.calculated)
    )
    # assert use and initial is not None, f"Initial=None for {causality}, {variability}"
    return use


initial_default = {
    #     default          possible values
    "A": (Initial.exact, (Initial.exact,)),
    "B": (Initial.calculated, (Initial.approx, Initial.calculated)),
    "C": (Initial.calculated, (Initial.exact, Initial.approx, Initial.calculated)),
    "D": (None, (None,)),
    "E": (None, (None,)),
}

explanations = {
    "a": """The combinations “constant / parameter”, “constant / calculatedParameter” and “constant / input” do not make sense,
                          since parameters and inputs are set from the environment, whereas a constant has always a value.""",
    "b": """The combinations “discrete / parameter”, “discrete / calculatedParameter”,
                          “continuous / parameter” and continuous / calculatedParameter do not make sense,
                          since causality = “parameter” and “calculatedParameter” define variables that do not depend on time,
                          whereas “discrete” and “continuous” define variables where the values can change during simulation.""",
    "c": """For an “independent” variable only variability = “continuous” makes sense.""",
    "d": """A fixed or tunable “input” has exactly the same properties as a fixed or tunable parameter.
                          For simplicity, only fixed and tunable parameters shall be defined.""",
    "e": """A fixed or tunable “output” has exactly the same properties as a fixed or tunable calculatedParameter.
                          For simplicity, only fixed and tunable calculatedParameters shall be defined.""",
}


def check_causality_variability_initial(
    causality: str | Enum | None,  # EnumType | None,
    variability: str | Enum | None,  # EnumType | None,
    initial: str | Enum | None,
) -> tuple[Causality | None, Variability | None, Initial | None]:
    _causality = ensure_enum(causality, Causality.parameter)
    _variability = ensure_enum(variability, Variability.constant)
    res = combination(_variability, _causality)  # type: ignore
    if res in ("a", "b", "c", "d", "e"):  # combination is not allowed
        logger.info(f"(causality {_causality}, variability {variability}) is not allowed: {explanations[res]}")
        return (None, None, None)
    else:  # allowed
        _initial = ensure_enum(initial, initial_default[res][0])
        if _initial not in initial_default[res][1]:
            logger.info(f"(Causality {_causality}, variability {_variability}, Initial {_initial}) is not allowed")
            return (None, None, None)
        if _initial is None:
            return (Causality(_causality), Variability(_variability), None)
        else:
            return (Causality(_causality), Variability(_variability), Initial(_initial))


class Check(IntFlag):
    """Flags to denote how variables should be checked with respect to units and range.
    The aspects are indepent, but can be combined in the Enum through | or &.

    * none:     neither units nor ranges are expected or checked.
    * unitNone: only numbers without units expected when new values are provided.
      If units are provided during initialization, these should be base units (SE), i.e. unit and display are the same.
    * u_all:    expect always quantity and number and convert internally to base units (SE). Provide output as display
    * units:    flag to filter only on units, e.g ck & Check.units
    * r_none:   no range is provided or checked
    * r_check:  range is provided and checked
    * ranges:  flag to filter on range, e.g. ck & Check.ranges
    * all:     short for u_all | r_check
    """

    none = 0
    u_none = 0
    u_all = 1
    units = 1
    r_none = 0
    r_check = 2
    ranges = 2
    all = 3
