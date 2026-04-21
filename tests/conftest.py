"""
Shared pytest fixtures for grbcompat tests.

grbcompat is imported directly (not via sys.modules patching) so that
test_gurobi_comparison.py can import the real gurobipy alongside the wrapper.
"""

import pytest

from grbcompat import GRB, Model
from grbcompat._expr import LinExpr


# ------------------------------------------------------------------ #
# Basic model fixtures
# ------------------------------------------------------------------ #

@pytest.fixture
def model():
    """Empty, silent model."""
    return Model("test")


@pytest.fixture
def xy_model():
    """Model with two continuous variables x, y (no constraints yet)."""
    m = Model("xy")
    x = m.addVar(lb=0.0, name="x")
    y = m.addVar(lb=0.0, name="y")
    return m, x, y


# ------------------------------------------------------------------ #
# Solved LP fixtures
# ------------------------------------------------------------------ #

@pytest.fixture
def solved_lp():
    """
    minimize  2x + 3y
    s.t.      x + y >= 4
              x     <= 6
              x, y  >= 0

    Optimal: x=4, y=0, obj=8.
    Dual of row 0 (x+y>=4) = 2.
    """
    m = Model("solved_lp")
    x = m.addVar(lb=0.0, name="x")
    y = m.addVar(lb=0.0, name="y")
    c0 = m.addConstr(x + y >= 4, name="demand")
    c1 = m.addConstr(x <= 6, name="cap")
    m.setObjective(2 * x + 3 * y, GRB.MINIMIZE)
    m.optimize()
    return m, x, y, c0, c1


@pytest.fixture
def solved_max_lp():
    """
    maximize  x + y
    s.t.      x + y <= 5
              x     <= 3
              y     <= 4
              x, y  >= 0

    Optimal: obj = 5 (e.g. x=1, y=4).
    """
    m = Model("max_lp")
    x = m.addVar(lb=0.0, name="x")
    y = m.addVar(lb=0.0, name="y")
    m.addConstr(x + y <= 5, name="sum_cap")
    m.addConstr(x <= 3, name="cap_x")
    m.addConstr(y <= 4, name="cap_y")
    m.setObjective(x + y, GRB.MAXIMIZE)
    m.optimize()
    return m, x, y
