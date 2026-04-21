"""Tests for Model.remove() – variable and constraint deletion."""

import pytest
from grbcompat import GRB, Model


TOL = 1e-6


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _simple_lp():
    """
    min  x + 2y + 3z
    s.t. x + y >= 1   (c0)
         y + z >= 1   (c1)
         x, y, z >= 0
    vars: x=col0, y=col1, z=col2
    """
    m = Model("simple")
    x = m.addVar(lb=0.0, name="x")
    y = m.addVar(lb=0.0, name="y")
    z = m.addVar(lb=0.0, name="z")
    c0 = m.addConstr(x + y >= 1, name="c0")
    c1 = m.addConstr(y + z >= 1, name="c1")
    m.setObjective(x + 2 * y + 3 * z, GRB.MINIMIZE)
    return m, x, y, z, c0, c1


# ------------------------------------------------------------------ #
# Single variable removal
# ------------------------------------------------------------------ #

class TestRemoveSingleVar:
    def test_num_vars_decrements(self):
        m, x, y, z, c0, c1 = _simple_lp()
        m.remove(y)
        assert m.NumVars == 2

    def test_getvars_excludes_removed(self):
        m, x, y, z, c0, c1 = _simple_lp()
        m.remove(y)
        assert not any(v is y for v in m.getVars())

    def test_getvars_includes_survivors(self):
        m, x, y, z, c0, c1 = _simple_lp()
        m.remove(y)
        assert any(v is x for v in m.getVars())
        assert any(v is z for v in m.getVars())

    def test_removed_var_col_idx_is_sentinel(self):
        m, x, y, z, c0, c1 = _simple_lp()
        m.remove(y)
        assert y._col_idx < 0

    def test_survivor_indices_recomputed_after_middle_removal(self):
        m, x, y, z, c0, c1 = _simple_lp()
        # Remove y (col 1); z was col 2, should become col 1.
        m.remove(y)
        assert x._col_idx == 0
        assert z._col_idx == 1

    def test_survivor_indices_unchanged_when_last_removed(self):
        m, x, y, z, c0, c1 = _simple_lp()
        m.remove(z)
        assert x._col_idx == 0
        assert y._col_idx == 1

    def test_solution_cleared_after_remove(self):
        m, x, y, z, c0, c1 = _simple_lp()
        m.optimize()
        assert m.Status == GRB.OPTIMAL
        m.remove(z)
        assert m.Status == GRB.LOADED

    def test_accessing_removed_var_X_raises(self):
        m, x, y, z, c0, c1 = _simple_lp()
        m.remove(y)
        with pytest.raises(RuntimeError):
            _ = y.X

    def test_accessing_removed_var_LB_raises(self):
        m, x, y, z, c0, c1 = _simple_lp()
        m.remove(y)
        with pytest.raises(RuntimeError):
            _ = y.LB

    def test_accessing_removed_var_UB_raises(self):
        m, x, y, z, c0, c1 = _simple_lp()
        m.remove(y)
        with pytest.raises(RuntimeError):
            _ = y.UB

    def test_accessing_removed_var_Obj_raises(self):
        m, x, y, z, c0, c1 = _simple_lp()
        m.remove(y)
        with pytest.raises(RuntimeError):
            _ = y.Obj

    def test_accessing_removed_var_VType_raises(self):
        m, x, y, z, c0, c1 = _simple_lp()
        m.remove(y)
        with pytest.raises(RuntimeError):
            _ = y.VType

    def test_var_name_still_readable_after_removal(self):
        """VarName is pure Python metadata and remains accessible."""
        m, x, y, z, c0, c1 = _simple_lp()
        m.remove(y)
        assert y.VarName == "y"

    def test_metadata_lists_shrink(self):
        m, x, y, z, c0, c1 = _simple_lp()
        m.remove(y)
        assert len(m._var_lbs) == 2
        assert len(m._var_ubs) == 2
        assert len(m._var_objs) == 2
        assert len(m._var_vtypes) == 2

    def test_remove_first_var(self):
        m, x, y, z, c0, c1 = _simple_lp()
        m.remove(x)
        assert m.NumVars == 2
        assert y._col_idx == 0
        assert z._col_idx == 1

    def test_remove_last_var(self):
        m, x, y, z, c0, c1 = _simple_lp()
        m.remove(z)
        assert m.NumVars == 2
        assert x._col_idx == 0
        assert y._col_idx == 1


# ------------------------------------------------------------------ #
# Single constraint removal
# ------------------------------------------------------------------ #

class TestRemoveSingleConstr:
    def test_num_constrs_decrements(self):
        m, x, y, z, c0, c1 = _simple_lp()
        m.remove(c0)
        assert m.NumConstrs == 1

    def test_getconstrs_excludes_removed(self):
        m, x, y, z, c0, c1 = _simple_lp()
        m.remove(c0)
        assert not any(c is c0 for c in m.getConstrs())
        assert any(c is c1 for c in m.getConstrs())

    def test_removed_constr_row_idx_is_sentinel(self):
        m, x, y, z, c0, c1 = _simple_lp()
        m.remove(c0)
        assert c0._row_idx < 0

    def test_survivor_row_index_recomputed(self):
        m, x, y, z, c0, c1 = _simple_lp()
        # c1 was row 1; after removing c0 (row 0) it becomes row 0.
        m.remove(c0)
        assert c1._row_idx == 0

    def test_accessing_removed_constr_Pi_raises(self):
        m, x, y, z, c0, c1 = _simple_lp()
        m.optimize()
        m.remove(c0)
        with pytest.raises(RuntimeError):
            _ = c0.Pi

    def test_accessing_removed_constr_Slack_raises(self):
        m, x, y, z, c0, c1 = _simple_lp()
        m.optimize()
        m.remove(c0)
        with pytest.raises(RuntimeError):
            _ = c0.Slack

    def test_setting_removed_constr_RHS_raises(self):
        m, x, y, z, c0, c1 = _simple_lp()
        m.remove(c0)
        with pytest.raises(RuntimeError):
            c0.RHS = 5.0

    def test_constr_name_still_readable_after_removal(self):
        m, x, y, z, c0, c1 = _simple_lp()
        m.remove(c0)
        assert c0.ConstrName == "c0"


# ------------------------------------------------------------------ #
# Batch removal
# ------------------------------------------------------------------ #

class TestRemoveBatch:
    def test_remove_var_list(self):
        m, x, y, z, c0, c1 = _simple_lp()
        m.remove([x, z])
        assert m.NumVars == 1
        assert y._col_idx == 0

    def test_remove_constr_list(self):
        m, x, y, z, c0, c1 = _simple_lp()
        m.remove([c0, c1])
        assert m.NumConstrs == 0

    def test_remove_mixed_list(self):
        m, x, y, z, c0, c1 = _simple_lp()
        m.remove([y, c1])
        assert m.NumVars == 2
        assert m.NumConstrs == 1
        assert x._col_idx == 0
        assert z._col_idx == 1
        assert c0._row_idx == 0

    def test_remove_non_contiguous_vars(self):
        m, x, y, z, c0, c1 = _simple_lp()
        m.remove([x, z])
        assert x._col_idx < 0
        assert z._col_idx < 0
        assert y._col_idx == 0

    def test_remove_non_contiguous_constrs(self):
        m = Model()
        v = m.addVar(lb=0.0, name="v")
        c0 = m.addConstr(v >= 1, name="c0")
        c1 = m.addConstr(v >= 2, name="c1")
        c2 = m.addConstr(v >= 3, name="c2")
        m.remove([c0, c2])
        assert c0._row_idx < 0
        assert c2._row_idx < 0
        assert c1._row_idx == 0


# ------------------------------------------------------------------ #
# Correctness after removal + re-solve
# ------------------------------------------------------------------ #

class TestRemoveAndResolve:
    def test_remove_var_reoptimize_correct_obj(self):
        """
        Original: min x + 2y + 3z  s.t. x+y>=1, y+z>=1
        Remove z. Remaining: min x + 2y  s.t. x+y>=1, y>=1
        Optimal: x=0, y=1, obj=2.
        """
        m, x, y, z, c0, c1 = _simple_lp()
        m.remove(z)
        m.optimize()
        assert m.Status == GRB.OPTIMAL
        assert m.ObjVal == pytest.approx(2.0, abs=TOL)

    def test_remove_constr_reoptimize_correct_obj(self):
        """
        Remove c1 (y+z>=1) from _simple_lp.
        Remaining: min x+2y+3z  s.t. x+y>=1
        Optimal: x=1, y=0, z=0, obj=1.
        """
        m, x, y, z, c0, c1 = _simple_lp()
        m.remove(c1)
        m.optimize()
        assert m.Status == GRB.OPTIMAL
        assert m.ObjVal == pytest.approx(1.0, abs=TOL)

    def test_survivor_var_values_correct_after_remove(self):
        """After removing z, survivor vars report correct primal values."""
        m, x, y, z, c0, c1 = _simple_lp()
        m.remove(z)
        m.optimize()
        assert x.X >= -TOL
        assert y.X >= -TOL

    def test_remove_middle_var_solution_is_correct(self):
        """
        Remove middle var y. Model becomes:
        min x + 3z  s.t. z >= 1  (c1 originally had y+z>=1, but y is gone)
        The constraint c0 (x+y>=1) also had y, so with y gone it becomes x>=1.
        Optimal: x=1, z=1, obj=4.
        """
        m, x, y, z, c0, c1 = _simple_lp()
        m.remove(y)
        m.optimize()
        assert m.Status == GRB.OPTIMAL
        assert x.X == pytest.approx(1.0, abs=TOL)
        assert z.X == pytest.approx(1.0, abs=TOL)
        assert m.ObjVal == pytest.approx(4.0, abs=TOL)

    def test_multiple_removes_then_solve(self):
        """Remove one var, then remove a constraint, then solve."""
        m, x, y, z, c0, c1 = _simple_lp()
        m.remove(z)
        m.remove(c1)
        # Remaining: min x + 2y  s.t. x+y>=1
        m.optimize()
        assert m.Status == GRB.OPTIMAL
        assert m.ObjVal == pytest.approx(1.0, abs=TOL)

    def test_duals_accessible_after_remove_and_resolve(self):
        m, x, y, z, c0, c1 = _simple_lp()
        m.remove(z)
        m.optimize()
        assert isinstance(c0.Pi, float)

    def test_remove_all_constrs_and_resolve(self):
        m, x, y, z, c0, c1 = _simple_lp()
        m.remove([c0, c1])
        m.optimize()
        # No constraints, minimizing non-negative vars → all zero, obj=0
        assert m.Status == GRB.OPTIMAL
        assert m.ObjVal == pytest.approx(0.0, abs=TOL)


# ------------------------------------------------------------------ #
# Sequential removals (index consistency across multiple remove calls)
# ------------------------------------------------------------------ #

class TestSequentialRemovals:
    def test_two_sequential_var_removes(self):
        m = Model()
        a = m.addVar(lb=0.0, name="a")
        b = m.addVar(lb=0.0, name="b")
        c = m.addVar(lb=0.0, name="c")
        d = m.addVar(lb=0.0, name="d")

        m.remove(b)  # a=col0, c=col1, d=col2
        assert a._col_idx == 0
        assert c._col_idx == 1
        assert d._col_idx == 2

        m.remove(c)  # a=col0, d=col1
        assert a._col_idx == 0
        assert d._col_idx == 1
        assert m.NumVars == 2

    def test_two_sequential_constr_removes(self):
        m = Model()
        v = m.addVar(lb=0.0)
        r0 = m.addConstr(v >= 0, name="r0")
        r1 = m.addConstr(v >= 1, name="r1")
        r2 = m.addConstr(v >= 2, name="r2")

        m.remove(r0)
        assert r1._row_idx == 0
        assert r2._row_idx == 1

        m.remove(r1)
        assert r2._row_idx == 0
        assert m.NumConstrs == 1

    def test_remove_solve_remove_solve(self):
        """Solve, remove a var, solve again, remove a constr, solve again."""
        m, x, y, z, c0, c1 = _simple_lp()

        m.optimize()
        assert m.Status == GRB.OPTIMAL

        m.remove(z)
        m.optimize()
        assert m.Status == GRB.OPTIMAL
        first_obj = m.ObjVal

        m.remove(c1)
        m.optimize()
        assert m.Status == GRB.OPTIMAL
        # Removing c1 relaxes the model → obj can only stay same or improve
        assert m.ObjVal <= first_obj + TOL
