"""
grbcompat
=================

A drop-in wrapper that lets gurobipy-based code run on HiGHS with minimal
changes.  Simply add one import **before** any gurobipy import:

    import grbcompat          # patches sys.modules['gurobipy']
    import gurobipy as gp             # now resolves to this wrapper
    from gurobipy import GRB          # works as expected

The wrapper exposes the same public names as gurobipy (Model, GRB, LinExpr,
quicksum, …) and translates them to the highspy / HiGHS solver under the hood.
"""

import sys

__version__ = "0.1.2"

from grbcompat._constants import GRB
from grbcompat._constr import Constr
from grbcompat._expr import LinExpr, TempConstr
from grbcompat._genconstr import GenConstr
from grbcompat._model import Model
from grbcompat._params import Params
from grbcompat._quadexpr import QuadExpr, TempQConstr
from grbcompat._sos import SOS
from grbcompat._tupledict import tupledict
from grbcompat._var import Var

__all__ = [
    "__version__",
    "GRB",
    "Model",
    "install",
    "Var",
    "Constr",
    "GenConstr",
    "SOS",
    "LinExpr",
    "TempConstr",
    "QuadExpr",
    "TempQConstr",
    "Params",
    "tupledict",
    "quicksum",
    "multidict",
    "GurobiError",
    "Env",
    "disposeDefaultEnv",
]


# ------------------------------------------------------------------ #
# gurobipy-compatible utilities
# ------------------------------------------------------------------ #

def quicksum(iterable) -> LinExpr:
    """
    Efficiently sum an iterable of Var / LinExpr / numeric objects into a
    single LinExpr, mirroring gurobipy.quicksum.
    """
    result = LinExpr()
    for item in iterable:
        result = result + item
    return result


def multidict(data: dict) -> list:
    """
    Decompose a dict whose values are lists into a list of dicts,
    mirroring gurobipy.multidict.

    Example::

        keys, cost, weight = gp.multidict({
            0: [1.0, 2.0],
            1: [3.0, 4.0],
        })
    """
    keys = list(data.keys())
    if not data:
        return [keys]
    first = next(iter(data.values()))
    if not isinstance(first, (list, tuple)):
        return [keys, data]
    n = len(first)
    dicts: list[dict] = [{} for _ in range(n)]
    for k, vals in data.items():
        for i, v in enumerate(vals):
            dicts[i][k] = v
    return [keys, *dicts]


class GurobiError(Exception):
    """Compatibility stub for gurobipy.GurobiError."""


class Env:
    """
    Compatibility stub for gurobipy.Env.

    gurobipy uses Env objects for licence management; this wrapper does not
    need them so all methods are no-ops.
    """

    def __init__(self, *args, **kwargs) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_) -> None:
        pass

    def start(self) -> None:
        pass

    def dispose(self) -> None:
        pass


def disposeDefaultEnv() -> None:
    """Compatibility stub for gurobipy.disposeDefaultEnv."""


# ------------------------------------------------------------------ #
# Explicit sys.modules patching (opt-in)
# ------------------------------------------------------------------ #

def install() -> None:
    """
    Patch ``sys.modules`` so that ``import gurobipy`` resolves to this
    wrapper for the remainder of the Python session.

    Call this **once**, before any ``import gurobipy`` statement, when you
    want existing gurobipy code to run on HiGHS with no further changes::

        import grbcompat
        grbcompat.install()          # <-- one line added

        import gurobipy as gp                # now backed by HiGHS
        from gurobipy import GRB
        # ... all existing code unchanged ...

    If you want to use both solvers in the same script, **do not** call
    ``install()``; instead import this package under its own name::

        import grbcompat as highs    # HiGHS solver
        import gurobipy as gp                # real Gurobi solver

        m_h = highs.Model()   # solved by HiGHS
        m_g = gp.Model()      # solved by Gurobi
    """
    sys.modules["gurobipy"] = sys.modules[__name__]
