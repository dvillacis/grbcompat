"""Tests for the Constr class."""

import pytest
from grbcompat import GRB, Model
from grbcompat._constr import Constr


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def simple_lp():
    """
    minimize  2x + 3y
    s.t.      x + y >= 4   (c_ge)
              x + y <= 6   (c_le)
              x == 2       (c_eq)
              x, y >= 0
    Optimal: x=2, y=2, obj=10.
    """
    m = Model()
    x = m.addVar(lb=0.0, name="x")
    y = m.addVar(lb=0.0, name="y")
    c_ge = m.addConstr(x + y >= 4, name="c_ge")
    c_le = m.addConstr(x + y <= 6, name="c_le")
    c_eq = m.addConstr(x == 2, name="c_eq")
    m.setObjective(2 * x + 3 * y, GRB.MINIMIZE)
    m.optimize()
    return m, x, y, c_ge, c_le, c_eq


# ------------------------------------------------------------------ #
# Construction and metadata
# ------------------------------------------------------------------ #

class TestConstrConstruction:
    def test_constr_returned_by_addconstr(self):
        m = Model()
        x = m.addVar(name="x")
        c = m.addConstr(x <= 5)
        assert isinstance(c, Constr)

    def test_constr_name(self):
        m = Model()
        x = m.addVar()
        c = m.addConstr(x <= 5, name="my_constr")
        assert c.ConstrName == "my_constr"

    def test_constrname_setter(self):
        m = Model()
        x = m.addVar()
        c = m.addConstr(x <= 5, name="old")
        c.ConstrName = "new"
        assert c.ConstrName == "new"

    def test_sense_le(self):
        m = Model()
        x = m.addVar()
        c = m.addConstr(x <= 5)
        assert c.Sense == GRB.LESS_EQUAL

    def test_sense_ge(self):
        m = Model()
        x = m.addVar()
        c = m.addConstr(x >= 2)
        assert c.Sense == GRB.GREATER_EQUAL

    def test_sense_eq(self):
        m = Model()
        x = m.addVar()
        c = m.addConstr(x == 3)
        assert c.Sense == GRB.EQUAL

    def test_rhs_le(self):
        m = Model()
        x = m.addVar()
        c = m.addConstr(x <= 7)
        assert c.RHS == pytest.approx(7.0)

    def test_rhs_ge(self):
        m = Model()
        x = m.addVar()
        c = m.addConstr(x >= 2)
        assert c.RHS == pytest.approx(2.0)

    def test_rhs_eq(self):
        m = Model()
        x = m.addVar()
        c = m.addConstr(x == 4)
        assert c.RHS == pytest.approx(4.0)

    def test_repr(self):
        m = Model()
        x = m.addVar()
        c = m.addConstr(x <= 5, name="cap")
        assert "cap" in repr(c)
        assert "<" in repr(c)


# ------------------------------------------------------------------ #
# Solution attributes
# ------------------------------------------------------------------ #

class TestConstrSolutionAttributes:
    def test_Pi_raises_before_optimize(self):
        m = Model()
        x = m.addVar()
        c = m.addConstr(x <= 5)
        with pytest.raises(RuntimeError):
            _ = c.Pi

    def test_Slack_raises_before_optimize(self):
        m = Model()
        x = m.addVar()
        c = m.addConstr(x <= 5)
        with pytest.raises(RuntimeError):
            _ = c.Slack

    def test_Pi_active_constraint(self):
        m, x, y, c_ge, c_le, c_eq = simple_lp()
        # c_eq (x==2) is active, should have nonzero dual
        assert isinstance(c_eq.Pi, float)

    def test_Pi_inactive_constraint(self):
        m, x, y, c_ge, c_le, c_eq = simple_lp()
        # c_le is x+y<=6, active value is 4, slack=2 → not active, Pi≈0
        assert c_le.Pi == pytest.approx(0.0, abs=1e-6)

    def test_Slack_zero_for_active_ge_constr(self):
        m, x, y, c_ge, c_le, c_eq = simple_lp()
        # c_ge: x+y>=4, x=2,y=2, activity=4 → slack=0
        assert c_ge.Slack == pytest.approx(0.0, abs=1e-6)

    def test_Slack_positive_for_inactive_le_constr(self):
        m, x, y, c_ge, c_le, c_eq = simple_lp()
        # c_le: x+y<=6, activity=4, rhs=6 → slack=2
        assert c_le.Slack == pytest.approx(2.0, abs=1e-6)

    def test_Slack_zero_for_eq_constr(self):
        m, x, y, c_ge, c_le, c_eq = simple_lp()
        assert c_eq.Slack == pytest.approx(0.0, abs=1e-6)

    def test_Pi_clears_after_reset(self):
        m, x, y, c_ge, c_le, c_eq = simple_lp()
        m.reset()
        with pytest.raises(RuntimeError):
            _ = c_ge.Pi


# ------------------------------------------------------------------ #
# RHS setter
# ------------------------------------------------------------------ #

class TestConstrRHSSetter:
    def test_rhs_setter_updates_stored_value(self):
        m = Model()
        x = m.addVar()
        c = m.addConstr(x <= 5)
        c.RHS = 10.0
        assert c.RHS == pytest.approx(10.0)

    def test_rhs_setter_takes_effect_at_optimize_le(self):
        m = Model()
        x = m.addVar(lb=0.0)
        c = m.addConstr(x <= 5)
        m.setObjective(x, GRB.MAXIMIZE)
        m.optimize()
        assert x.X == pytest.approx(5.0)
        c.RHS = 8.0
        m.optimize()
        assert x.X == pytest.approx(8.0)

    def test_rhs_setter_takes_effect_at_optimize_ge(self):
        m = Model()
        x = m.addVar(lb=0.0)
        c = m.addConstr(x >= 2)
        m.setObjective(x, GRB.MINIMIZE)
        m.optimize()
        assert x.X == pytest.approx(2.0)
        c.RHS = 5.0
        m.optimize()
        assert x.X == pytest.approx(5.0)

    def test_rhs_setter_takes_effect_at_optimize_eq(self):
        m = Model()
        x = m.addVar(lb=0.0)
        c = m.addConstr(x == 3)
        m.setObjective(x, GRB.MINIMIZE)
        m.optimize()
        assert x.X == pytest.approx(3.0)
        c.RHS = 7.0
        m.optimize()
        assert x.X == pytest.approx(7.0)
