"""Tests for Model.copy()."""

import pytest
from grbcompat import GRB, Model


TOL = 1e-6


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _build_lp():
    """
    min  2x + 3y
    s.t. x + y >= 4   (c0, GE)
         x     <= 6   (c1, LE)
         x, y >= 0
    Optimal: x=4, y=0, obj=8.
    """
    m = Model("orig")
    x = m.addVar(lb=0.0, name="x")
    y = m.addVar(lb=0.0, name="y")
    c0 = m.addConstr(x + y >= 4, name="demand")
    c1 = m.addConstr(x <= 6, name="cap")
    m.setObjective(2 * x + 3 * y, GRB.MINIMIZE)
    return m, x, y, c0, c1


def _build_mip():
    """
    max  2*b0 + b1 + z
    s.t. b0 + b1 <= 1   (at most one binary selected)
         z <= 3          (integer cap)
         b0, b1 in {0,1}; z integer >= 0
    Optimal: b0=1, b1=0, z=3, obj=5.
    """
    m = Model("mip")
    b0 = m.addVar(vtype=GRB.BINARY, name="b0")
    b1 = m.addVar(vtype=GRB.BINARY, name="b1")
    z  = m.addVar(lb=0.0, vtype=GRB.INTEGER, name="z")
    m.addConstr(b0 + b1 <= 1, name="c0")
    m.addConstr(z <= 3, name="c1")
    m.setObjective(2 * b0 + b1 + z, GRB.MAXIMIZE)
    return m, b0, b1, z


# ------------------------------------------------------------------ #
# Basic copy properties
# ------------------------------------------------------------------ #

class TestCopyBasic:
    def test_returns_model(self):
        m, *_ = _build_lp()
        assert isinstance(m.copy(), Model)

    def test_different_object(self):
        m, *_ = _build_lp()
        assert m.copy() is not m

    def test_same_num_vars(self):
        m, *_ = _build_lp()
        assert m.copy().NumVars == m.NumVars

    def test_same_num_constrs(self):
        m, *_ = _build_lp()
        assert m.copy().NumConstrs == m.NumConstrs

    def test_model_name_preserved(self):
        m, *_ = _build_lp()
        assert m.copy().ModelName == "orig"

    def test_copy_starts_unsolved(self):
        m, *_ = _build_lp()
        assert m.copy().Status == GRB.LOADED

    def test_copy_of_solved_model_starts_unsolved(self):
        m, *_ = _build_lp()
        m.optimize()
        assert m.copy().Status == GRB.LOADED

    def test_empty_model_copy(self):
        m = Model("empty")
        cp = m.copy()
        assert cp.NumVars == 0
        assert cp.NumConstrs == 0


# ------------------------------------------------------------------ #
# Variable metadata preserved
# ------------------------------------------------------------------ #

class TestCopyVarMetadata:
    def setup_method(self):
        self.m, self.x, self.y, self.c0, self.c1 = _build_lp()
        self.cp = self.m.copy()
        self.cx, self.cy = self.cp.getVars()

    def test_var_lb_preserved(self):
        assert self.cx.LB == pytest.approx(self.x.LB)
        assert self.cy.LB == pytest.approx(self.y.LB)

    def test_var_ub_preserved(self):
        assert self.cx.UB == pytest.approx(self.x.UB)
        assert self.cy.UB == pytest.approx(self.y.UB)

    def test_var_obj_coeff_preserved(self):
        # Var.Obj reflects the coefficient passed to addVar(obj=...) or set
        # via var.Obj = ..., not the coefficient set by setObjective().
        # The _build_lp fixture uses addVar defaults (obj=0) then setObjective.
        assert self.cx.Obj == pytest.approx(self.x.Obj)
        assert self.cy.Obj == pytest.approx(self.y.Obj)

    def test_var_name_preserved(self):
        assert self.cx.VarName == "x"
        assert self.cy.VarName == "y"

    def test_var_vtype_preserved(self):
        assert self.cx.VType == GRB.CONTINUOUS
        assert self.cy.VType == GRB.CONTINUOUS

    def test_var_objects_are_new(self):
        assert self.cx is not self.x
        assert self.cy is not self.y

    def test_var_col_indices_match(self):
        assert self.cx._col_idx == self.x._col_idx
        assert self.cy._col_idx == self.y._col_idx


# ------------------------------------------------------------------ #
# MIP variable types preserved
# ------------------------------------------------------------------ #

class TestCopyMIPVarTypes:
    def setup_method(self):
        self.m, self.b0, self.b1, self.z = _build_mip()
        cp = self.m.copy()
        self.cb0, self.cb1, self.cz = cp.getVars()

    def test_binary_type_preserved_b0(self):
        assert self.cb0.VType == GRB.BINARY

    def test_binary_type_preserved_b1(self):
        assert self.cb1.VType == GRB.BINARY

    def test_integer_type_preserved(self):
        assert self.cz.VType == GRB.INTEGER

    def test_binary_bounds_preserved(self):
        assert self.cb0.LB == pytest.approx(0.0)
        assert self.cb0.UB == pytest.approx(1.0)


# ------------------------------------------------------------------ #
# Constraint metadata preserved
# ------------------------------------------------------------------ #

class TestCopyConstrMetadata:
    def setup_method(self):
        self.m, self.x, self.y, self.c0, self.c1 = _build_lp()
        cp = self.m.copy()
        self.cc0, self.cc1 = cp.getConstrs()

    def test_constr_name_preserved(self):
        assert self.cc0.ConstrName == "demand"
        assert self.cc1.ConstrName == "cap"

    def test_constr_sense_preserved(self):
        assert self.cc0.Sense == GRB.GREATER_EQUAL
        assert self.cc1.Sense == GRB.LESS_EQUAL

    def test_constr_rhs_preserved(self):
        assert self.cc0.RHS == pytest.approx(4.0)
        assert self.cc1.RHS == pytest.approx(6.0)

    def test_constr_objects_are_new(self):
        assert self.cc0 is not self.c0
        assert self.cc1 is not self.c1

    def test_constr_row_indices_match(self):
        assert self.cc0._row_idx == self.c0._row_idx
        assert self.cc1._row_idx == self.c1._row_idx


# ------------------------------------------------------------------ #
# Objective preserved
# ------------------------------------------------------------------ #

class TestCopyObjective:
    def test_minimize_sense_preserved(self):
        m, *_ = _build_lp()
        assert m.copy().ModelSense == GRB.MINIMIZE

    def test_maximize_sense_preserved(self):
        m = Model()
        x = m.addVar(lb=0.0, ub=5.0)
        m.setObjective(x, GRB.MAXIMIZE)
        assert m.copy().ModelSense == GRB.MAXIMIZE

    def test_objective_offset_preserved(self):
        """ObjCon (offset) should survive the copy."""
        m = Model()
        x = m.addVar(lb=2.0, ub=2.0)
        m.setObjective(x + 10, GRB.MINIMIZE)
        cp = m.copy()
        cp.optimize()
        assert cp.ObjVal == pytest.approx(12.0, abs=TOL)


# ------------------------------------------------------------------ #
# Solve correctness
# ------------------------------------------------------------------ #

class TestCopySolve:
    def test_copy_solves_to_same_obj(self):
        m, *_ = _build_lp()
        m.optimize()
        cp = m.copy()
        cp.optimize()
        assert cp.ObjVal == pytest.approx(m.ObjVal, abs=TOL)

    def test_copy_optimal_status(self):
        m, *_ = _build_lp()
        cp = m.copy()
        cp.optimize()
        assert cp.Status == GRB.OPTIMAL

    def test_copy_primal_values_correct(self):
        m, *_ = _build_lp()
        cp = m.copy()
        cp.optimize()
        cx, cy = cp.getVars()
        assert cx.X == pytest.approx(4.0, abs=TOL)
        assert cy.X == pytest.approx(0.0, abs=TOL)

    def test_copy_duals_accessible_after_solve(self):
        m, *_ = _build_lp()
        cp = m.copy()
        cp.optimize()
        cc0, cc1 = cp.getConstrs()
        assert isinstance(cc0.Pi, float)

    def test_mip_copy_solves_correctly(self):
        m, *_ = _build_mip()
        cp = m.copy()
        cp.optimize()
        assert cp.Status == GRB.OPTIMAL
        assert cp.ObjVal == pytest.approx(5.0, abs=TOL)


# ------------------------------------------------------------------ #
# Independence: changes to one model do not affect the other
# ------------------------------------------------------------------ #

class TestCopyIndependence:
    def test_modify_original_bound_does_not_affect_copy(self):
        m, x, y, c0, c1 = _build_lp()
        cp = m.copy()
        x.UB = 1.0  # tighten bound on original
        # copy's var still has the old (infinite) upper bound
        cx = cp.getVars()[0]
        assert cx.UB > 1.0 + TOL

    def test_modify_copy_bound_does_not_affect_original(self):
        m, x, y, c0, c1 = _build_lp()
        cp = m.copy()
        cx = cp.getVars()[0]
        cx.UB = 1.0
        assert x.UB > 1.0 + TOL

    def test_original_solve_does_not_affect_copy_status(self):
        m, *_ = _build_lp()
        cp = m.copy()
        m.optimize()
        assert cp.Status == GRB.LOADED

    def test_copy_solve_does_not_affect_original_status(self):
        m, *_ = _build_lp()
        m.optimize()
        cp = m.copy()
        cp.optimize()
        # re-check original status is still what it was
        assert m.Status == GRB.OPTIMAL

    def test_remove_var_from_copy_does_not_affect_original(self):
        m, x, y, c0, c1 = _build_lp()
        cp = m.copy()
        cy = cp.getVars()[1]
        cp.remove(cy)
        assert m.NumVars == 2

    def test_remove_var_from_original_does_not_affect_copy(self):
        m, x, y, c0, c1 = _build_lp()
        cp = m.copy()
        m.remove(y)
        assert cp.NumVars == 2

    def test_solve_original_copy_independently_same_obj(self):
        m, *_ = _build_lp()
        cp = m.copy()
        m.optimize()
        cp.optimize()
        assert m.ObjVal == pytest.approx(cp.ObjVal, abs=TOL)

    def test_copy_rhs_change_independent(self):
        m, x, y, c0, c1 = _build_lp()
        cp = m.copy()
        cc0 = cp.getConstrs()[0]
        cc0.RHS = 10.0
        # original constraint unchanged
        assert c0.RHS == pytest.approx(4.0)
        # copy re-solves with tighter constraint
        cp.optimize()
        m.optimize()
        assert cp.ObjVal > m.ObjVal - TOL
