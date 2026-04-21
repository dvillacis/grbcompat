"""Params proxy – maps gurobipy parameter names to HiGHS options."""

from __future__ import annotations


# Maps gurobipy parameter name → (highs_option_name, type_coercion)
_PARAM_MAP: dict[str, tuple[str | None, type | None]] = {
    "TimeLimit": ("time_limit", float),
    "MIPGap": ("mip_rel_gap", float),
    "MIPGapAbs": ("mip_abs_gap", float),
    "FeasibilityTol": ("primal_feasibility_tolerance", float),
    "OptimalityTol": ("dual_feasibility_tolerance", float),
    "IntFeasTol": ("mip_feasibility_tolerance", float),
    "OutputFlag": ("output_flag", bool),
    "Threads": ("threads", int),
    "Seed": ("random_seed", int),
    "NodeLimit": ("mip_max_nodes", int),
    "IterationLimit": ("simplex_iteration_limit", int),
    "LogFile": ("log_file", str),
    "Presolve": ("presolve", str),
    # Silently ignored – no equivalent in HiGHS
    "SolFiles": (None, None),
    "SolutionNumber": (None, None),
    "LazyConstraints": (None, None),   # wrapper handles lazy constraints in Python
    "Method": (None, None),            # Gurobi LP/MIP algorithm selector, no HiGHS equivalent
}


class Params:
    """
    Proxy object accessible as ``model.Params``.

    Attribute assignment maps gurobipy parameter names to their HiGHS
    equivalents::

        model.Params.TimeLimit = 60
        model.Params.MIPGap = 0.01
        model.Params.OutputFlag = 1
    """

    def __init__(self, highs_instance) -> None:
        object.__setattr__(self, "_h", highs_instance)

    def __setattr__(self, name: str, value) -> None:
        h = object.__getattribute__(self, "_h")
        entry = _PARAM_MAP.get(name)
        if entry is not None:
            highs_name, coerce = entry
            if highs_name is None:
                return  # silently skip unsupported params
            h.setOptionValue(highs_name, coerce(value))  # type: ignore[misc]
        else:
            # Pass unknown names straight through to HiGHS
            import highspy as _hs
            status = h.setOptionValue(name, value)
            if status == _hs.HighsStatus.kError:
                raise AttributeError(f"Unknown Gurobi/HiGHS parameter: '{name}'")

    def __getattr__(self, name: str):
        h = object.__getattribute__(self, "_h")
        entry = _PARAM_MAP.get(name)
        if entry is not None:
            highs_name, _ = entry
            if highs_name is None:
                return None
            ok, val = h.getOptionValue(highs_name)
            if ok:
                return val
        try:
            ok, val = h.getOptionValue(name)
            if ok:
                return val
        except Exception:
            pass
        raise AttributeError(f"Unknown parameter: '{name}'")
