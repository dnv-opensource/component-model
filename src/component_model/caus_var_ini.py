import logging
from enum import Enum, EnumType

# import pythonfmu.enums # type: ignore
from pythonfmu.enums import Fmi2Causality as Causality  # type: ignore
from pythonfmu.enums import Fmi2Initial as Initial  # type: ignore
from pythonfmu.enums import Fmi2Variability as Variability  # type: ignore

logger = logging.getLogger(__name__)


# class Initial(Enum):
#     exact = 0
#     approx = 1
#     calculated = 2
#     none = 3  # additional value to allow for the cases when initial: --
#

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


def ensure_enum(org: str | Enum | None, typ: EnumType, default: Enum | None) -> Enum | None:
    """Ensure that we have an Enum, based on the input as str, Enum or None."""
    if org is None:
        if typ is Causality or typ is Variability:
            assert default is not None, "default value needed at this stage"
            return default
        elif typ is Initial:
            return default  # might be None
        else:
            return None
    elif isinstance(org, str):
        assert isinstance(typ, EnumType), f"EnumType expected as typ. Found {typ}"
        try:
            return typ[org]  # type: ignore [reportReturnType] # do not understand the error message
        except KeyError as err:
            raise Exception(f"The value {org} is not compatible with the Enum {typ}: {err}") from err
    else:
        assert isinstance(org, typ), f"{org} is not member of the Enum {typ}"
        return org


def check_causality_variability_initial(
    causality: str | EnumType | None,
    variability: str | EnumType | None,
    initial: str | Enum | None,
) -> tuple[Causality | None, Variability | None, Initial | None]:
    _causality = ensure_enum(causality, Causality, Causality.parameter)  # type: ignore
    _variability = ensure_enum(variability, Variability, Variability.constant)  # type: ignore
    res = combination(_variability, _causality)  # type: ignore
    if res in ("a", "b", "c", "d", "e"):  # combination is not allowed
        logger.info(f"(causality {_causality}, variability {variability}) is not allowed: {explanations[res]}")
        return (None, None, None)
    else:  # allowed
        _initial = ensure_enum(initial, Initial, initial_default[res][0])  # type: ignore
        if _initial not in initial_default[res][1]:
            logger.info(f"(Causality {_causality}, variability {_variability}, Initial {_initial}) is not allowed")
            return (None, None, None)
        if _initial is None:
            return (Causality(_causality), Variability(_variability), None)
        else:
            return (Causality(_causality), Variability(_variability), Initial(_initial))
