"""Tests for general constraints: addGenConstrIndicator / Abs / Min / Max."""

import pytest
from grbcompat import GRB, Model
from grbcompat._genconstr import GenConstr


TOL = 1e-5


# ================================================================== #
# addGenConstrIndicator
# ================================================================== #

class TestIndicatorBasic:
    def test_returns_genconstr(self):
        m = Model()
        b = m.addVar(vtype=GRB.BINARY)
        x = m.addVar(lb=0.0, ub=10.0)
        gc = m.addGenConstrIndicator(b, 1, x <= 5)
        assert isinstance(gc, GenConstr)

    def test_type_is_indicator(self):
        m = Model()
        b = m.addVar(vtype=GRB.BINARY)
        x = m.addVar(lb=0.0, ub=10.0)
        gc = m.addGenConstrIndicator(b, 1, x <= 5)
        assert gc._type == GenConstr.INDICATOR

    def test_name_stored(self):
        m = Model()
        b = m.addVar(vtype=GRB.BINARY)
        x = m.addVar(lb=0.0, ub=10.0)
        gc = m.addGenConstrIndicator(b, 1, x <= 5, name="ind1")
        assert gc._name == "ind1"

    def test_repr_contains_indicator(self):
        m = Model()
        b = m.addVar(vtype=GRB.BINARY)
        x = m.addVar(lb=0.0, ub=10.0)
        gc = m.addGenConstrIndicator(b, 1, x <= 5)
        assert "INDICATOR" in repr(gc)

    def test_numgenconstrs_increments(self):
        m = Model()
        b = m.addVar(vtype=GRB.BINARY)
        x = m.addVar(lb=0.0, ub=10.0)
        assert m.NumGenConstrs == 0
        m.addGenConstrIndicator(b, 1, x <= 5)
        assert m.NumGenConstrs == 1

    def test_getgenconstrs_returns_list(self):
        m = Model()
        b = m.addVar(vtype=GRB.BINARY)
        x = m.addVar(lb=0.0, ub=10.0)
        gc = m.addGenConstrIndicator(b, 1, x <= 5)
        result = m.getGenConstrs()
        assert isinstance(result, list)
        assert result[0] is gc

    def test_le_adds_one_helper_constr(self):
        m = Model()
        b = m.addVar(vtype=GRB.BINARY)
        x = m.addVar(lb=0.0, ub=10.0)
        gc = m.addGenConstrIndicator(b, 1, x <= 5)
        assert len(gc._helper_constrs) == 1

    def test_ge_adds_one_helper_constr(self):
        m = Model()
        b = m.addVar(vtype=GRB.BINARY)
        x = m.addVar(lb=0.0, ub=10.0)
        gc = m.addGenConstrIndicator(b, 1, x >= 3)
        assert len(gc._helper_constrs) == 1

    def test_eq_adds_two_helper_constrs(self):
        m = Model()
        b = m.addVar(vtype=GRB.BINARY)
        x = m.addVar(lb=0.0, ub=10.0)
        gc = m.addGenConstrIndicator(b, 1, x == 4)
        assert len(gc._helper_constrs) == 2

    def test_no_helper_vars(self):
        """Indicator constraints need no new variables."""
        m = Model()
        b = m.addVar(vtype=GRB.BINARY)
        x = m.addVar(lb=0.0, ub=10.0)
        gc = m.addGenConstrIndicator(b, 1, x <= 5)
        assert gc._helper_vars == []

    def test_rhs_variable_raises(self):
        m = Model()
        b = m.addVar(vtype=GRB.BINARY)
        x = m.addVar(lb=0.0, ub=10.0)
        y = m.addVar(lb=0.0, ub=10.0)
        with pytest.raises(ValueError):
            m.addGenConstrIndicator(b, 1, x, GRB.LESS_EQUAL, y)

    def test_explicit_sense_rhs(self):
        """addGenConstrIndicator also accepts (lhs, sense, rhs) form."""
        m = Model()
        b = m.addVar(vtype=GRB.BINARY)
        x = m.addVar(lb=0.0, ub=10.0)
        gc = m.addGenConstrIndicator(b, 1, x, GRB.LESS_EQUAL, 5)
        assert isinstance(gc, GenConstr)


class TestIndicatorCorrectness:
    def test_binval1_le_active(self):
        """b=1 → x <= 5, b forced to 1: x must be <= 5."""
        m = Model()
        b = m.addVar(vtype=GRB.BINARY, name="b")
        x = m.addVar(lb=0.0, ub=10.0, name="x")
        m.addConstr(b == 1)
        m.addGenConstrIndicator(b, 1, x <= 5)
        m.setObjective(x, GRB.MAXIMIZE)
        m.optimize()
        assert m.Status == GRB.OPTIMAL
        assert x.X == pytest.approx(5.0, abs=TOL)

    def test_binval1_le_inactive(self):
        """b=1 → x <= 5, b forced to 0: x can reach its UB."""
        m = Model()
        b = m.addVar(vtype=GRB.BINARY, name="b")
        x = m.addVar(lb=0.0, ub=10.0, name="x")
        m.addConstr(b == 0)
        m.addGenConstrIndicator(b, 1, x <= 5)
        m.setObjective(x, GRB.MAXIMIZE)
        m.optimize()
        assert m.Status == GRB.OPTIMAL
        assert x.X == pytest.approx(10.0, abs=TOL)

    def test_binval0_le_active(self):
        """b=0 → x <= 5, b forced to 0: x must be <= 5."""
        m = Model()
        b = m.addVar(vtype=GRB.BINARY, name="b")
        x = m.addVar(lb=0.0, ub=10.0, name="x")
        m.addConstr(b == 0)
        m.addGenConstrIndicator(b, 0, x <= 5)
        m.setObjective(x, GRB.MAXIMIZE)
        m.optimize()
        assert m.Status == GRB.OPTIMAL
        assert x.X == pytest.approx(5.0, abs=TOL)

    def test_binval1_ge_active(self):
        """b=1 → x >= 3, b forced to 1: objective min x → x = 3."""
        m = Model()
        b = m.addVar(vtype=GRB.BINARY)
        x = m.addVar(lb=0.0, ub=10.0)
        m.addConstr(b == 1)
        m.addGenConstrIndicator(b, 1, x >= 3)
        m.setObjective(x, GRB.MINIMIZE)
        m.optimize()
        assert m.Status == GRB.OPTIMAL
        assert x.X == pytest.approx(3.0, abs=TOL)

    def test_binval1_ge_inactive(self):
        """b=1 → x >= 3, b forced to 0: min x → x = 0."""
        m = Model()
        b = m.addVar(vtype=GRB.BINARY)
        x = m.addVar(lb=0.0, ub=10.0)
        m.addConstr(b == 0)
        m.addGenConstrIndicator(b, 1, x >= 3)
        m.setObjective(x, GRB.MINIMIZE)
        m.optimize()
        assert m.Status == GRB.OPTIMAL
        assert x.X == pytest.approx(0.0, abs=TOL)

    def test_binval1_eq_active(self):
        """b=1 → x == 4, b forced to 1: x must be 4."""
        m = Model()
        b = m.addVar(vtype=GRB.BINARY)
        x = m.addVar(lb=0.0, ub=10.0)
        m.addConstr(b == 1)
        m.addGenConstrIndicator(b, 1, x == 4)
        m.setObjective(x, GRB.MAXIMIZE)
        m.optimize()
        assert m.Status == GRB.OPTIMAL
        assert x.X == pytest.approx(4.0, abs=TOL)

    def test_multi_term_lhs(self):
        """b=1 → x + y <= 7, b forced to 1, max x+y → x+y = 7."""
        m = Model()
        b = m.addVar(vtype=GRB.BINARY)
        x = m.addVar(lb=0.0, ub=10.0)
        y = m.addVar(lb=0.0, ub=10.0)
        m.addConstr(b == 1)
        m.addGenConstrIndicator(b, 1, x + y <= 7)
        m.setObjective(x + y, GRB.MAXIMIZE)
        m.optimize()
        assert m.Status == GRB.OPTIMAL
        assert (x.X + y.X) == pytest.approx(7.0, abs=TOL)

    def test_indicator_selects_branch(self):
        """
        Two branches controlled by binary b:
          b=1 → x = 2
          b=0 → x = 8
        Objective min x → b=1 optimal.
        """
        m = Model()
        b = m.addVar(vtype=GRB.BINARY)
        x = m.addVar(lb=0.0, ub=10.0)
        m.addGenConstrIndicator(b, 1, x == 2)
        m.addGenConstrIndicator(b, 0, x == 8)
        m.setObjective(x, GRB.MINIMIZE)
        m.optimize()
        assert m.Status == GRB.OPTIMAL
        assert b.X == pytest.approx(1.0, abs=TOL)
        assert x.X == pytest.approx(2.0, abs=TOL)


class TestIndicatorIndex:
    def test_first_index_is_zero(self):
        m = Model()
        b = m.addVar(vtype=GRB.BINARY)
        x = m.addVar(lb=0.0, ub=10.0)
        gc = m.addGenConstrIndicator(b, 1, x <= 5)
        assert gc.index == 0

    def test_second_index_is_one(self):
        m = Model()
        b = m.addVar(vtype=GRB.BINARY)
        x = m.addVar(lb=0.0, ub=10.0)
        gc0 = m.addGenConstrIndicator(b, 1, x <= 5)
        gc1 = m.addGenConstrIndicator(b, 0, x >= 2)
        assert gc1.index == 1

    def test_removed_raises(self):
        m = Model()
        b = m.addVar(vtype=GRB.BINARY)
        x = m.addVar(lb=0.0, ub=10.0)
        gc = m.addGenConstrIndicator(b, 1, x <= 5)
        m.remove(gc)
        with pytest.raises(RuntimeError):
            _ = gc.index


class TestIndicatorRemove:
    def test_numgenconstrs_decrements(self):
        m = Model()
        b = m.addVar(vtype=GRB.BINARY)
        x = m.addVar(lb=0.0, ub=10.0)
        gc = m.addGenConstrIndicator(b, 1, x <= 5)
        m.remove(gc)
        assert m.NumGenConstrs == 0

    def test_helper_constrs_removed(self):
        m = Model()
        b = m.addVar(vtype=GRB.BINARY)
        x = m.addVar(lb=0.0, ub=10.0)
        gc = m.addGenConstrIndicator(b, 1, x <= 5)
        n_before = m.NumConstrs
        m.remove(gc)
        assert m.NumConstrs == n_before - 1

    def test_remove_then_constraint_not_enforced(self):
        """After removing the indicator, x can exceed 5 even with b=1."""
        m = Model()
        b = m.addVar(vtype=GRB.BINARY)
        x = m.addVar(lb=0.0, ub=10.0)
        m.addConstr(b == 1)
        gc = m.addGenConstrIndicator(b, 1, x <= 5)
        m.remove(gc)
        m.setObjective(x, GRB.MAXIMIZE)
        m.optimize()
        assert m.Status == GRB.OPTIMAL
        assert x.X == pytest.approx(10.0, abs=TOL)

    def test_getgenconstrs_empty_after_remove(self):
        m = Model()
        b = m.addVar(vtype=GRB.BINARY)
        x = m.addVar(lb=0.0, ub=10.0)
        gc = m.addGenConstrIndicator(b, 1, x <= 5)
        m.remove(gc)
        assert m.getGenConstrs() == []


class TestIndicatorCopy:
    def test_copy_preserves_numgenconstrs(self):
        m = Model()
        b = m.addVar(vtype=GRB.BINARY)
        x = m.addVar(lb=0.0, ub=10.0)
        m.addGenConstrIndicator(b, 1, x <= 5)
        cp = m.copy()
        assert cp.NumGenConstrs == 1

    def test_copy_solves_correctly(self):
        m = Model()
        b = m.addVar(vtype=GRB.BINARY)
        x = m.addVar(lb=0.0, ub=10.0)
        m.addConstr(b == 1)
        m.addGenConstrIndicator(b, 1, x <= 5)
        m.setObjective(x, GRB.MAXIMIZE)
        cp = m.copy()
        cp.optimize()
        assert cp.Status == GRB.OPTIMAL
        cp_vars = cp.getVars()
        assert cp_vars[1].X == pytest.approx(5.0, abs=TOL)

    def test_copy_is_independent(self):
        m = Model()
        b = m.addVar(vtype=GRB.BINARY)
        x = m.addVar(lb=0.0, ub=10.0)
        gc = m.addGenConstrIndicator(b, 1, x <= 5)
        cp = m.copy()
        cp.remove(cp.getGenConstrs()[0])
        assert m.NumGenConstrs == 1
        assert cp.NumGenConstrs == 0


# ================================================================== #
# addGenConstrAbs
# ================================================================== #

class TestAbsBasic:
    def test_returns_genconstr(self):
        m = Model()
        r = m.addVar(lb=0.0, ub=10.0, name="r")
        a = m.addVar(lb=-5.0, ub=5.0, name="a")
        gc = m.addGenConstrAbs(r, a)
        assert isinstance(gc, GenConstr)

    def test_type_is_abs(self):
        m = Model()
        r = m.addVar(lb=0.0, ub=10.0)
        a = m.addVar(lb=-5.0, ub=5.0)
        gc = m.addGenConstrAbs(r, a)
        assert gc._type == GenConstr.ABS

    def test_one_binary_helper_var(self):
        m = Model()
        r = m.addVar(lb=0.0, ub=10.0)
        a = m.addVar(lb=-5.0, ub=5.0)
        gc = m.addGenConstrAbs(r, a)
        assert len(gc._helper_vars) == 1

    def test_four_helper_constrs(self):
        m = Model()
        r = m.addVar(lb=0.0, ub=10.0)
        a = m.addVar(lb=-5.0, ub=5.0)
        gc = m.addGenConstrAbs(r, a)
        assert len(gc._helper_constrs) == 4

    def test_numgenconstrs_increments(self):
        m = Model()
        r = m.addVar(lb=0.0, ub=10.0)
        a = m.addVar(lb=-5.0, ub=5.0)
        assert m.NumGenConstrs == 0
        m.addGenConstrAbs(r, a)
        assert m.NumGenConstrs == 1


class TestAbsCorrectness:
    def test_positive_argvar(self):
        """argvar fixed to 3 (positive): resvar should be 3."""
        m = Model()
        r = m.addVar(lb=0.0, ub=10.0, name="r")
        a = m.addVar(lb=3.0, ub=3.0, name="a")
        m.addGenConstrAbs(r, a)
        m.setObjective(r, GRB.MINIMIZE)
        m.optimize()
        assert m.Status == GRB.OPTIMAL
        assert r.X == pytest.approx(3.0, abs=TOL)

    def test_negative_argvar(self):
        """argvar fixed to -4 (negative): resvar should be 4."""
        m = Model()
        r = m.addVar(lb=0.0, ub=10.0, name="r")
        a = m.addVar(lb=-4.0, ub=-4.0, name="a")
        m.addGenConstrAbs(r, a)
        m.setObjective(r, GRB.MINIMIZE)
        m.optimize()
        assert m.Status == GRB.OPTIMAL
        assert r.X == pytest.approx(4.0, abs=TOL)

    def test_zero_argvar(self):
        """argvar = 0: resvar should be 0."""
        m = Model()
        r = m.addVar(lb=0.0, ub=10.0, name="r")
        a = m.addVar(lb=0.0, ub=0.0, name="a")
        m.addGenConstrAbs(r, a)
        m.setObjective(r, GRB.MINIMIZE)
        m.optimize()
        assert m.Status == GRB.OPTIMAL
        assert r.X == pytest.approx(0.0, abs=TOL)

    def test_minimize_sum_forces_abs(self):
        """
        min r + x, r = |x|, x in [-5, 5].
        Optimal x=0, r=0, obj=0.
        """
        m = Model()
        r = m.addVar(lb=0.0, ub=10.0, name="r")
        x = m.addVar(lb=-5.0, ub=5.0, name="x")
        m.addGenConstrAbs(r, x)
        m.setObjective(r + x, GRB.MINIMIZE)
        m.optimize()
        assert m.Status == GRB.OPTIMAL
        assert r.X == pytest.approx(abs(x.X), abs=TOL)

    def test_abs_in_mip_context(self):
        """
        r = |x|, x integer in [-5, 5], minimize r.
        Optimal: x=0, r=0.
        """
        m = Model()
        r = m.addVar(lb=0.0, ub=10.0, name="r")
        x = m.addVar(lb=-5.0, ub=5.0, vtype=GRB.INTEGER, name="x")
        m.addGenConstrAbs(r, x)
        m.setObjective(r, GRB.MINIMIZE)
        m.optimize()
        assert m.Status == GRB.OPTIMAL
        assert r.X == pytest.approx(0.0, abs=TOL)


class TestAbsRemove:
    def test_numgenconstrs_decrements(self):
        m = Model()
        r = m.addVar(lb=0.0, ub=10.0)
        a = m.addVar(lb=-5.0, ub=5.0)
        gc = m.addGenConstrAbs(r, a)
        m.remove(gc)
        assert m.NumGenConstrs == 0

    def test_helper_vars_removed(self):
        m = Model()
        r = m.addVar(lb=0.0, ub=10.0)
        a = m.addVar(lb=-5.0, ub=5.0)
        n_before = m.NumVars
        gc = m.addGenConstrAbs(r, a)
        m.remove(gc)
        assert m.NumVars == n_before  # the 1 binary helper is gone


class TestAbsCopy:
    def test_copy_preserves_numgenconstrs(self):
        m = Model()
        r = m.addVar(lb=0.0, ub=10.0)
        a = m.addVar(lb=-5.0, ub=5.0)
        m.addGenConstrAbs(r, a)
        cp = m.copy()
        assert cp.NumGenConstrs == 1

    def test_copy_solves_correctly(self):
        m = Model()
        r = m.addVar(lb=0.0, ub=10.0)
        a = m.addVar(lb=3.0, ub=3.0)
        m.addGenConstrAbs(r, a)
        m.setObjective(r, GRB.MINIMIZE)
        cp = m.copy()
        cp.optimize()
        assert cp.Status == GRB.OPTIMAL
        assert cp.getVars()[0].X == pytest.approx(3.0, abs=TOL)


# ================================================================== #
# addGenConstrMin
# ================================================================== #

class TestMinBasic:
    def test_returns_genconstr(self):
        m = Model()
        r = m.addVar(lb=0.0, ub=10.0)
        x = m.addVar(lb=0.0, ub=10.0)
        y = m.addVar(lb=0.0, ub=10.0)
        gc = m.addGenConstrMin(r, [x, y])
        assert isinstance(gc, GenConstr)

    def test_type_is_min(self):
        m = Model()
        r = m.addVar(lb=0.0, ub=10.0)
        x = m.addVar(lb=0.0, ub=10.0)
        gc = m.addGenConstrMin(r, [x])
        assert gc._type == GenConstr.MIN

    def test_numgenconstrs_increments(self):
        m = Model()
        r = m.addVar(lb=0.0, ub=10.0)
        x = m.addVar(lb=0.0, ub=10.0)
        assert m.NumGenConstrs == 0
        m.addGenConstrMin(r, [x])
        assert m.NumGenConstrs == 1


class TestMinCorrectness:
    def test_two_vars_selects_minimum(self):
        """
        r = min(x, y); x=3, y=7 (fixed), max r → r=3.
        """
        m = Model()
        r = m.addVar(lb=0.0, ub=10.0, name="r")
        x = m.addVar(lb=3.0, ub=3.0, name="x")
        y = m.addVar(lb=7.0, ub=7.0, name="y")
        m.addGenConstrMin(r, [x, y])
        m.setObjective(r, GRB.MAXIMIZE)
        m.optimize()
        assert m.Status == GRB.OPTIMAL
        assert r.X == pytest.approx(3.0, abs=TOL)

    def test_three_vars_selects_minimum(self):
        """r = min(2, 5, 9) → r = 2."""
        m = Model()
        r = m.addVar(lb=0.0, ub=10.0, name="r")
        x0 = m.addVar(lb=2.0, ub=2.0)
        x1 = m.addVar(lb=5.0, ub=5.0)
        x2 = m.addVar(lb=9.0, ub=9.0)
        m.addGenConstrMin(r, [x0, x1, x2])
        m.setObjective(r, GRB.MAXIMIZE)
        m.optimize()
        assert m.Status == GRB.OPTIMAL
        assert r.X == pytest.approx(2.0, abs=TOL)

    def test_with_constant_wins(self):
        """r = min(x, 4); x=8 (fixed), constant wins: r=4."""
        m = Model()
        r = m.addVar(lb=0.0, ub=10.0, name="r")
        x = m.addVar(lb=8.0, ub=8.0, name="x")
        m.addGenConstrMin(r, [x], constant=4.0)
        m.setObjective(r, GRB.MAXIMIZE)
        m.optimize()
        assert m.Status == GRB.OPTIMAL
        assert r.X == pytest.approx(4.0, abs=TOL)

    def test_with_constant_var_wins(self):
        """r = min(x, 10); x=3 (fixed), var wins: r=3."""
        m = Model()
        r = m.addVar(lb=0.0, ub=10.0, name="r")
        x = m.addVar(lb=3.0, ub=3.0, name="x")
        m.addGenConstrMin(r, [x], constant=10.0)
        m.setObjective(r, GRB.MAXIMIZE)
        m.optimize()
        assert m.Status == GRB.OPTIMAL
        assert r.X == pytest.approx(3.0, abs=TOL)

    def test_min_in_objective(self):
        """
        max r, r = min(x, y), x+y <= 10, x,y in [0, 10].
        Best: x=y=5, r=5.
        """
        m = Model()
        r = m.addVar(lb=0.0, ub=10.0, name="r")
        x = m.addVar(lb=0.0, ub=10.0, name="x")
        y = m.addVar(lb=0.0, ub=10.0, name="y")
        m.addConstr(x + y <= 10)
        m.addGenConstrMin(r, [x, y])
        m.setObjective(r, GRB.MAXIMIZE)
        m.optimize()
        assert m.Status == GRB.OPTIMAL
        assert r.X == pytest.approx(5.0, abs=TOL)


class TestMinRemoveCopy:
    def test_remove_decrements_numgenconstrs(self):
        m = Model()
        r = m.addVar(lb=0.0, ub=10.0)
        x = m.addVar(lb=0.0, ub=10.0)
        y = m.addVar(lb=0.0, ub=10.0)
        gc = m.addGenConstrMin(r, [x, y])
        m.remove(gc)
        assert m.NumGenConstrs == 0

    def test_copy_preserves_numgenconstrs(self):
        m = Model()
        r = m.addVar(lb=0.0, ub=10.0)
        x = m.addVar(lb=0.0, ub=10.0)
        y = m.addVar(lb=0.0, ub=10.0)
        m.addGenConstrMin(r, [x, y])
        cp = m.copy()
        assert cp.NumGenConstrs == 1


# ================================================================== #
# addGenConstrMax
# ================================================================== #

class TestMaxBasic:
    def test_returns_genconstr(self):
        m = Model()
        r = m.addVar(lb=0.0, ub=10.0)
        x = m.addVar(lb=0.0, ub=10.0)
        gc = m.addGenConstrMax(r, [x])
        assert isinstance(gc, GenConstr)

    def test_type_is_max(self):
        m = Model()
        r = m.addVar(lb=0.0, ub=10.0)
        x = m.addVar(lb=0.0, ub=10.0)
        gc = m.addGenConstrMax(r, [x])
        assert gc._type == GenConstr.MAX

    def test_numgenconstrs_increments(self):
        m = Model()
        r = m.addVar(lb=0.0, ub=10.0)
        x = m.addVar(lb=0.0, ub=10.0)
        assert m.NumGenConstrs == 0
        m.addGenConstrMax(r, [x])
        assert m.NumGenConstrs == 1


class TestMaxCorrectness:
    def test_two_vars_selects_maximum(self):
        """r = max(x, y); x=3, y=7 (fixed), min r → r=7."""
        m = Model()
        r = m.addVar(lb=0.0, ub=10.0, name="r")
        x = m.addVar(lb=3.0, ub=3.0, name="x")
        y = m.addVar(lb=7.0, ub=7.0, name="y")
        m.addGenConstrMax(r, [x, y])
        m.setObjective(r, GRB.MINIMIZE)
        m.optimize()
        assert m.Status == GRB.OPTIMAL
        assert r.X == pytest.approx(7.0, abs=TOL)

    def test_three_vars_selects_maximum(self):
        """r = max(2, 5, 9) → r = 9."""
        m = Model()
        r = m.addVar(lb=0.0, ub=10.0, name="r")
        x0 = m.addVar(lb=2.0, ub=2.0)
        x1 = m.addVar(lb=5.0, ub=5.0)
        x2 = m.addVar(lb=9.0, ub=9.0)
        m.addGenConstrMax(r, [x0, x1, x2])
        m.setObjective(r, GRB.MINIMIZE)
        m.optimize()
        assert m.Status == GRB.OPTIMAL
        assert r.X == pytest.approx(9.0, abs=TOL)

    def test_with_constant_wins(self):
        """r = max(x, 10); x=2 (fixed), constant wins: r=10."""
        m = Model()
        r = m.addVar(lb=0.0, ub=10.0, name="r")
        x = m.addVar(lb=2.0, ub=2.0, name="x")
        m.addGenConstrMax(r, [x], constant=10.0)
        m.setObjective(r, GRB.MINIMIZE)
        m.optimize()
        assert m.Status == GRB.OPTIMAL
        assert r.X == pytest.approx(10.0, abs=TOL)

    def test_with_constant_var_wins(self):
        """r = max(x, 3); x=8 (fixed), var wins: r=8."""
        m = Model()
        r = m.addVar(lb=0.0, ub=10.0, name="r")
        x = m.addVar(lb=8.0, ub=8.0, name="x")
        m.addGenConstrMax(r, [x], constant=3.0)
        m.setObjective(r, GRB.MINIMIZE)
        m.optimize()
        assert m.Status == GRB.OPTIMAL
        assert r.X == pytest.approx(8.0, abs=TOL)

    def test_max_in_objective(self):
        """
        min r, r = max(x, y), x+y >= 6, x,y in [0, 10].
        Best: x=y=3, r=3.
        """
        m = Model()
        r = m.addVar(lb=0.0, ub=10.0, name="r")
        x = m.addVar(lb=0.0, ub=10.0, name="x")
        y = m.addVar(lb=0.0, ub=10.0, name="y")
        m.addConstr(x + y >= 6)
        m.addGenConstrMax(r, [x, y])
        m.setObjective(r, GRB.MINIMIZE)
        m.optimize()
        assert m.Status == GRB.OPTIMAL
        assert r.X == pytest.approx(3.0, abs=TOL)


class TestMaxRemoveCopy:
    def test_remove_decrements_numgenconstrs(self):
        m = Model()
        r = m.addVar(lb=0.0, ub=10.0)
        x = m.addVar(lb=0.0, ub=10.0)
        gc = m.addGenConstrMax(r, [x])
        m.remove(gc)
        assert m.NumGenConstrs == 0

    def test_copy_preserves_numgenconstrs(self):
        m = Model()
        r = m.addVar(lb=0.0, ub=10.0)
        x = m.addVar(lb=0.0, ub=10.0)
        m.addGenConstrMax(r, [x])
        cp = m.copy()
        assert cp.NumGenConstrs == 1

    def test_copy_solves_correctly(self):
        m = Model()
        r = m.addVar(lb=0.0, ub=10.0)
        x = m.addVar(lb=3.0, ub=3.0)
        y = m.addVar(lb=7.0, ub=7.0)
        m.addGenConstrMax(r, [x, y])
        m.setObjective(r, GRB.MINIMIZE)
        cp = m.copy()
        cp.optimize()
        assert cp.Status == GRB.OPTIMAL
        assert cp.getVars()[0].X == pytest.approx(7.0, abs=TOL)

    def test_copy_is_independent(self):
        m = Model()
        r = m.addVar(lb=0.0, ub=10.0)
        x = m.addVar(lb=0.0, ub=10.0)
        m.addGenConstrMax(r, [x])
        cp = m.copy()
        cp.remove(cp.getGenConstrs()[0])
        assert m.NumGenConstrs == 1
        assert cp.NumGenConstrs == 0
