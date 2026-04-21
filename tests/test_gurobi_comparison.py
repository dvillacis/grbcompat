"""
Cross-solver comparison tests: grbcompat (HiGHS) vs real Gurobi.

These tests build identical models with both solvers and verify that the
results (objective value, primal solution, duals, status) agree within a
reasonable tolerance.

Requirements
------------
* ``gurobipy`` must be installed (``pip install gurobipy``).
* A valid Gurobi licence must be active on this machine.

All tests in this module are automatically **skipped** when either
requirement is not met — no configuration change is needed.

Dual-solver pattern
-------------------
This module demonstrates the intended side-by-side usage::

    import grbcompat as highs   # backed by HiGHS (free)
    import gurobipy as gp               # backed by Gurobi (licensed)

    m_h = highs.Model(...)   # solved by HiGHS
    m_g = gp.Model(...)      # solved by Gurobi

``grbcompat.install()`` is intentionally NOT called here so that
``import gurobipy`` returns the real Gurobi package.
"""

import pytest

import grbcompat as highs
from grbcompat import GRB as HGRB, quicksum as hquicksum

# ------------------------------------------------------------------ #
# Guard: skip entire module if Gurobi is unavailable / unlicensed
# ------------------------------------------------------------------ #

try:
    import gurobipy as _grb
    from gurobipy import GRB as _GRB, quicksum as _gquicksum

    # Verify a licence is present by creating a silent model
    _check = _grb.Model()
    _check.setParam("OutputFlag", 0)
    _check.dispose()
    _GUROBI_OK = True
except Exception:
    _GUROBI_OK = False

pytestmark = pytest.mark.skipif(
    not _GUROBI_OK,
    reason="Gurobi is not installed or no valid licence found",
)

# Tolerances for cross-solver comparison
OBJ_TOL = 1e-4   # relative tolerance for objective values
SOL_TOL = 1e-4   # absolute tolerance for primal/dual values


# ------------------------------------------------------------------ #
# Model-builder helpers
# The same function signature is used for both solvers so we can call
# it with either (highs, HGRB) or (_grb, _GRB) and get the same model.
# ------------------------------------------------------------------ #

def _silent(m):
    """Suppress solver output regardless of which solver is used."""
    try:
        m.setParam("OutputFlag", 0)  # gurobipy
    except Exception:
        pass  # wrapper silences HiGHS in __init__
    return m


def build_simple_lp(Mod, G):
    """
    minimize  2x + 3y
    s.t.      x + y >= 4
              x     <= 6
              x, y  >= 0
    Optimal: x=4, y=0, obj=8.
    """
    m = _silent(Mod())
    x = m.addVar(lb=0.0, name="x")
    y = m.addVar(lb=0.0, name="y")
    c0 = m.addConstr(x + y >= 4, name="demand")
    c1 = m.addConstr(x <= 6, name="cap")
    m.setObjective(2 * x + 3 * y, G.MINIMIZE)
    m.optimize()
    return m, x, y, c0, c1


def build_max_lp(Mod, G):
    """
    maximize  x + y
    s.t.      x + y <= 5
              x     <= 3
              y     <= 4
              x, y  >= 0
    Optimal: obj=5.
    """
    m = _silent(Mod())
    x = m.addVar(lb=0.0, name="x")
    y = m.addVar(lb=0.0, name="y")
    m.addConstr(x + y <= 5)
    m.addConstr(x <= 3)
    m.addConstr(y <= 4)
    m.setObjective(x + y, G.MAXIMIZE)
    m.optimize()
    return m, x, y


def build_equality_lp(Mod, G):
    """
    minimize  x + y
    s.t.      x + y == 5
              x, y  >= 0
    Optimal: obj=5.
    """
    m = _silent(Mod())
    x = m.addVar(lb=0.0, name="x")
    y = m.addVar(lb=0.0, name="y")
    m.addConstr(x + y == 5)
    m.setObjective(x + y, G.MINIMIZE)
    m.optimize()
    return m, x, y


def build_obj_constant_lp(Mod, G):
    """
    minimize  x + 7
    s.t.      x >= 2
    Optimal: obj=9.
    """
    m = _silent(Mod())
    x = m.addVar(lb=0.0, name="x")
    m.addConstr(x >= 2)
    m.setObjective(x + 7, G.MINIMIZE)
    m.optimize()
    return m, x


def build_infeasible_lp(Mod, G):
    m = _silent(Mod())
    x = m.addVar(lb=0.0, name="x")
    m.addConstr(x >= 5)
    m.addConstr(x <= 3)
    m.setObjective(x, G.MINIMIZE)
    m.optimize()
    return m


def build_knapsack(Mod, G, Qs):
    """0-1 knapsack, optimal obj=18."""
    values  = [10, 6, 4, 5, 3]
    weights = [ 5, 4, 3, 2, 1]
    capacity = 8
    n = len(values)
    m = _silent(Mod())
    x = m.addVars(n, vtype=G.BINARY, name="x")
    m.setObjective(Qs(values[i] * x[i] for i in range(n)), G.MAXIMIZE)
    m.addConstr(Qs(weights[i] * x[i] for i in range(n)) <= capacity)
    m.optimize()
    return m, x


def build_integer_lp(Mod, G):
    """
    minimize  x + y
    s.t.      x + y >= 2.5
              x, y  >= 0, integer
    Optimal: obj=3.
    """
    m = _silent(Mod())
    x = m.addVar(lb=0.0, vtype=G.INTEGER, name="x")
    y = m.addVar(lb=0.0, vtype=G.INTEGER, name="y")
    m.addConstr(x + y >= 2.5)
    m.setObjective(x + y, G.MINIMIZE)
    m.optimize()
    return m, x, y


def build_transport(Mod, G, Qs):
    """2×3 transportation LP."""
    supply = [30, 40]
    demand = [20, 25, 25]
    cost = [[2, 3, 1], [5, 4, 8]]
    m = _silent(Mod())
    x = m.addVars(2, 3, lb=0.0, name="x")
    m.setObjective(
        Qs(cost[i][j] * x[i, j] for i in range(2) for j in range(3)),
        G.MINIMIZE,
    )
    for i in range(2):
        m.addConstr(Qs(x[i, j] for j in range(3)) <= supply[i])
    for j in range(3):
        m.addConstr(Qs(x[i, j] for i in range(2)) >= demand[j])
    m.optimize()
    return m, x


# ------------------------------------------------------------------ #
# LP objective comparison
# ------------------------------------------------------------------ #

class TestLPObjectiveComparison:
    def test_minimize_obj(self):
        m_h, *_ = build_simple_lp(highs.Model, HGRB)
        m_g, *_ = build_simple_lp(_grb.Model, _GRB)
        assert m_h.ObjVal == pytest.approx(m_g.ObjVal, rel=OBJ_TOL)

    def test_maximize_obj(self):
        m_h, *_ = build_max_lp(highs.Model, HGRB)
        m_g, *_ = build_max_lp(_grb.Model, _GRB)
        assert m_h.ObjVal == pytest.approx(m_g.ObjVal, rel=OBJ_TOL)

    def test_equality_constraint_obj(self):
        m_h, *_ = build_equality_lp(highs.Model, HGRB)
        m_g, *_ = build_equality_lp(_grb.Model, _GRB)
        assert m_h.ObjVal == pytest.approx(m_g.ObjVal, rel=OBJ_TOL)

    def test_obj_constant_term(self):
        m_h, _ = build_obj_constant_lp(highs.Model, HGRB)
        m_g, _ = build_obj_constant_lp(_grb.Model, _GRB)
        assert m_h.ObjVal == pytest.approx(m_g.ObjVal, rel=OBJ_TOL)

    def test_transport_obj(self):
        m_h, _ = build_transport(highs.Model, HGRB, hquicksum)
        m_g, _ = build_transport(_grb.Model, _GRB, _gquicksum)
        assert m_h.ObjVal == pytest.approx(m_g.ObjVal, rel=OBJ_TOL)


# ------------------------------------------------------------------ #
# LP primal solution comparison
# ------------------------------------------------------------------ #

class TestLPPrimalComparison:
    def test_minimize_x_value(self):
        m_h, x_h, y_h, c0_h, c1_h = build_simple_lp(highs.Model, HGRB)
        m_g, x_g, y_g, c0_g, c1_g = build_simple_lp(_grb.Model, _GRB)
        assert x_h.X == pytest.approx(x_g.X, abs=SOL_TOL)

    def test_minimize_y_value(self):
        m_h, x_h, y_h, c0_h, c1_h = build_simple_lp(highs.Model, HGRB)
        m_g, x_g, y_g, c0_g, c1_g = build_simple_lp(_grb.Model, _GRB)
        assert y_h.X == pytest.approx(y_g.X, abs=SOL_TOL)

    def test_equality_constr_feasibility(self):
        m_h, x_h, y_h = build_equality_lp(highs.Model, HGRB)
        m_g, x_g, y_g = build_equality_lp(_grb.Model, _GRB)
        assert x_h.X + y_h.X == pytest.approx(5.0, abs=SOL_TOL)
        assert x_g.X + y_g.X == pytest.approx(5.0, abs=SOL_TOL)

    def test_maximize_obj_value_of_primal(self):
        m_h, x_h, y_h = build_max_lp(highs.Model, HGRB)
        m_g, x_g, y_g = build_max_lp(_grb.Model, _GRB)
        assert x_h.X + y_h.X == pytest.approx(x_g.X + y_g.X, abs=SOL_TOL)

    def test_obj_constant_primal_x(self):
        m_h, x_h = build_obj_constant_lp(highs.Model, HGRB)
        m_g, x_g = build_obj_constant_lp(_grb.Model, _GRB)
        assert x_h.X == pytest.approx(x_g.X, abs=SOL_TOL)


# ------------------------------------------------------------------ #
# LP dual comparison
# ------------------------------------------------------------------ #

class TestLPDualComparison:
    def test_binding_constraint_pi(self):
        """Both solvers must agree on the shadow price of a binding constraint."""
        m_h, x_h, y_h, c0_h, c1_h = build_simple_lp(highs.Model, HGRB)
        m_g, x_g, y_g, c0_g, c1_g = build_simple_lp(_grb.Model, _GRB)
        assert c0_h.Pi == pytest.approx(c0_g.Pi, abs=SOL_TOL)

    def test_nonbinding_constraint_pi(self):
        """Both solvers must agree that a non-binding constraint has Pi ≈ 0."""
        m_h, x_h, y_h, c0_h, c1_h = build_simple_lp(highs.Model, HGRB)
        m_g, x_g, y_g, c0_g, c1_g = build_simple_lp(_grb.Model, _GRB)
        assert c1_h.Pi == pytest.approx(c1_g.Pi, abs=SOL_TOL)

    def test_binding_constraint_slack_zero(self):
        m_h, x_h, y_h, c0_h, c1_h = build_simple_lp(highs.Model, HGRB)
        m_g, x_g, y_g, c0_g, c1_g = build_simple_lp(_grb.Model, _GRB)
        assert c0_h.Slack == pytest.approx(c0_g.Slack, abs=SOL_TOL)

    def test_nonbinding_constraint_slack(self):
        m_h, x_h, y_h, c0_h, c1_h = build_simple_lp(highs.Model, HGRB)
        m_g, x_g, y_g, c0_g, c1_g = build_simple_lp(_grb.Model, _GRB)
        assert c1_h.Slack == pytest.approx(c1_g.Slack, abs=SOL_TOL)

    def test_equality_constraint_pi(self):
        m_h = _silent(highs.Model())
        m_g = _silent(_grb.Model())
        x_h = m_h.addVar(lb=0.0)
        y_h = m_h.addVar(lb=0.0)
        x_g = m_g.addVar(lb=0.0)
        y_g = m_g.addVar(lb=0.0)
        c_h = m_h.addConstr(x_h + y_h == 5)
        c_g = m_g.addConstr(x_g + y_g == 5)
        m_h.setObjective(x_h + 2 * y_h, HGRB.MINIMIZE)
        m_g.setObjective(x_g + 2 * y_g, _GRB.MINIMIZE)
        m_h.optimize()
        m_g.optimize()
        assert c_h.Pi == pytest.approx(c_g.Pi, abs=SOL_TOL)


# ------------------------------------------------------------------ #
# Status comparison
# ------------------------------------------------------------------ #

class TestStatusComparison:
    def test_both_report_optimal(self):
        m_h, *_ = build_simple_lp(highs.Model, HGRB)
        m_g, *_ = build_simple_lp(_grb.Model, _GRB)
        assert m_h.Status == HGRB.OPTIMAL
        assert m_g.Status == _GRB.OPTIMAL
        assert m_h.Status == m_g.Status

    def test_both_report_infeasible(self):
        m_h = build_infeasible_lp(highs.Model, HGRB)
        m_g = build_infeasible_lp(_grb.Model, _GRB)
        assert m_h.Status == HGRB.INFEASIBLE
        assert m_g.Status == _GRB.INFEASIBLE
        assert m_h.Status == m_g.Status

    def test_maximize_both_optimal(self):
        m_h, *_ = build_max_lp(highs.Model, HGRB)
        m_g, *_ = build_max_lp(_grb.Model, _GRB)
        assert m_h.Status == HGRB.OPTIMAL
        assert m_g.Status == _GRB.OPTIMAL


# ------------------------------------------------------------------ #
# MIP objective comparison
# ------------------------------------------------------------------ #

class TestMIPComparison:
    def test_knapsack_obj(self):
        m_h, _ = build_knapsack(highs.Model, HGRB, hquicksum)
        m_g, _ = build_knapsack(_grb.Model, _GRB, _gquicksum)
        assert m_h.ObjVal == pytest.approx(m_g.ObjVal, rel=OBJ_TOL)

    def test_knapsack_status(self):
        m_h, _ = build_knapsack(highs.Model, HGRB, hquicksum)
        m_g, _ = build_knapsack(_grb.Model, _GRB, _gquicksum)
        assert m_h.Status == HGRB.OPTIMAL
        assert m_g.Status == _GRB.OPTIMAL

    def test_knapsack_solution_feasibility(self):
        weights = [5, 4, 3, 2, 1]
        capacity = 8
        _, x_h = build_knapsack(highs.Model, HGRB, hquicksum)
        _, x_g = build_knapsack(_grb.Model, _GRB, _gquicksum)
        w_h = sum(weights[i] * x_h[i].X for i in range(5))
        w_g = sum(weights[i] * x_g[i].X for i in range(5))
        assert w_h <= capacity + SOL_TOL
        assert w_g <= capacity + SOL_TOL

    def test_integer_program_obj(self):
        m_h, *_ = build_integer_lp(highs.Model, HGRB)
        m_g, *_ = build_integer_lp(_grb.Model, _GRB)
        assert m_h.ObjVal == pytest.approx(m_g.ObjVal, rel=OBJ_TOL)

    def test_integer_program_integrality(self):
        _, x_h, y_h = build_integer_lp(highs.Model, HGRB)
        _, x_g, y_g = build_integer_lp(_grb.Model, _GRB)
        for v in (x_h, y_h, x_g, y_g):
            assert abs(v.X - round(v.X)) < SOL_TOL

    def test_transport_feasibility(self):
        demand = [20, 25, 25]
        _, x_h = build_transport(highs.Model, HGRB, hquicksum)
        _, x_g = build_transport(_grb.Model, _GRB, _gquicksum)
        for j in range(3):
            flow_h = sum(x_h[i, j].X for i in range(2))
            flow_g = sum(x_g[i, j].X for i in range(2))
            assert flow_h >= demand[j] - SOL_TOL
            assert flow_g >= demand[j] - SOL_TOL


# ------------------------------------------------------------------ #
# Dual-solver in same script (illustrative)
# ------------------------------------------------------------------ #

class TestDualSolverInSameScript:
    """
    Demonstrates that both solvers can be used simultaneously in the
    same Python process without any conflict.
    """

    def test_same_objective_from_both_solvers(self):
        # HiGHS model
        m_h = _silent(highs.Model("dual_highs"))
        x_h = m_h.addVar(lb=0.0)
        y_h = m_h.addVar(lb=0.0)
        m_h.addConstr(x_h + y_h >= 3)
        m_h.setObjective(x_h + y_h, HGRB.MINIMIZE)
        m_h.optimize()

        # Gurobi model (same problem)
        m_g = _silent(_grb.Model("dual_gurobi"))
        x_g = m_g.addVar(lb=0.0)
        y_g = m_g.addVar(lb=0.0)
        m_g.addConstr(x_g + y_g >= 3)
        m_g.setObjective(x_g + y_g, _GRB.MINIMIZE)
        m_g.optimize()

        assert m_h.Status == HGRB.OPTIMAL
        assert m_g.Status == _GRB.OPTIMAL
        assert m_h.ObjVal == pytest.approx(m_g.ObjVal, rel=OBJ_TOL)

    def test_independent_models_do_not_interfere(self):
        """HiGHS model with a different objective should not affect Gurobi model."""
        m_h = _silent(highs.Model())
        x_h = m_h.addVar(lb=5.0, ub=5.0)   # forced to 5
        m_h.setObjective(x_h, HGRB.MINIMIZE)
        m_h.optimize()

        m_g = _silent(_grb.Model())
        x_g = m_g.addVar(lb=10.0, ub=10.0)  # forced to 10
        m_g.setObjective(x_g, _GRB.MINIMIZE)
        m_g.optimize()

        assert x_h.X == pytest.approx(5.0, abs=SOL_TOL)
        assert x_g.X == pytest.approx(10.0, abs=SOL_TOL)
        assert m_h.ObjVal != pytest.approx(m_g.ObjVal)  # intentionally different

    def test_gurobi_module_is_not_the_wrapper(self):
        """Confirm real Gurobi is used, not the wrapper."""
        import gurobipy as real_gp
        assert real_gp is not highs, (
            "gurobipy resolved to grbcompat – "
            "did install() get called somewhere? Check test order."
        )

    def test_both_solvers_report_same_status_for_infeasible(self):
        m_h = build_infeasible_lp(highs.Model, HGRB)
        m_g = build_infeasible_lp(_grb.Model, _GRB)
        assert m_h.Status == HGRB.INFEASIBLE
        assert m_g.Status == _GRB.INFEASIBLE

    def test_highs_grb_constants_match_gurobi_constants(self):
        """Key numeric constants must agree between the two GRB namespaces."""
        assert HGRB.MINIMIZE == _GRB.MINIMIZE
        assert HGRB.MAXIMIZE == _GRB.MAXIMIZE
        assert HGRB.OPTIMAL  == _GRB.OPTIMAL
        assert HGRB.INFEASIBLE == _GRB.INFEASIBLE
        assert HGRB.CONTINUOUS == _GRB.CONTINUOUS
        assert HGRB.BINARY == _GRB.BINARY
        assert HGRB.INTEGER == _GRB.INTEGER
        assert HGRB.LESS_EQUAL == _GRB.LESS_EQUAL
        assert HGRB.GREATER_EQUAL == _GRB.GREATER_EQUAL
        assert HGRB.EQUAL == _GRB.EQUAL
