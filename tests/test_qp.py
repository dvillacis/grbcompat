"""Tests for quadratic programming (QuadExpr, QP objective, copy, addQConstr stub)."""

from __future__ import annotations

import math
import os
import tempfile
import pytest

import grbcompat as gp
from grbcompat import GRB, LinExpr, QuadExpr


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def fresh_model():
    m = gp.Model()
    m.Params.OutputFlag = 0
    return m


# ------------------------------------------------------------------ #
# TestQuadExpr – expression construction and arithmetic
# ------------------------------------------------------------------ #

class TestQuadExpr:

    def test_var_times_var_returns_quadexpr(self):
        m = fresh_model()
        x = m.addVar(lb=-gp.GRB.INFINITY)
        y = m.addVar(lb=-gp.GRB.INFINITY)
        q = x * y
        assert isinstance(q, QuadExpr)
        assert q.size() == 1
        assert q.getCoeff(0) == 1.0
        assert q.getVar1(0) is x
        assert q.getVar2(0) is y

    def test_var_times_self(self):
        m = fresh_model()
        x = m.addVar()
        q = x * x
        assert isinstance(q, QuadExpr)
        assert q.size() == 1
        assert q.getCoeff(0) == 1.0
        assert q.getVar1(0) is x
        assert q.getVar2(0) is x

    def test_quadexpr_size_and_accessors(self):
        m = fresh_model()
        x = m.addVar()
        y = m.addVar()
        q = x * x + 3.0 * (y * y)
        assert q.size() == 2

    def test_scalar_times_quadexpr(self):
        m = fresh_model()
        x = m.addVar()
        q = 5.0 * (x * x)
        assert isinstance(q, QuadExpr)
        assert q.size() == 1
        assert math.isclose(q.getCoeff(0), 5.0)

    def test_quadexpr_times_scalar(self):
        m = fresh_model()
        x = m.addVar()
        q = (x * x) * 3.0
        assert math.isclose(q.getCoeff(0), 3.0)

    def test_quadexpr_div_scalar(self):
        m = fresh_model()
        x = m.addVar()
        q = (x * x) / 2.0
        assert math.isclose(q.getCoeff(0), 0.5)

    def test_neg_quadexpr(self):
        m = fresh_model()
        x = m.addVar()
        q = -(x * x)
        assert math.isclose(q.getCoeff(0), -1.0)

    def test_quadexpr_add_quadexpr(self):
        m = fresh_model()
        x = m.addVar()
        y = m.addVar()
        q = (x * x) + (y * y)
        assert isinstance(q, QuadExpr)
        assert q.size() == 2

    def test_quadexpr_sub_quadexpr(self):
        m = fresh_model()
        x = m.addVar()
        q = (x * x) - (x * x)
        assert isinstance(q, QuadExpr)
        # Two terms with opposite signs remain (merging is done at Hessian build)
        assert q.size() == 2

    def test_quadexpr_add_linexpr(self):
        m = fresh_model()
        x = m.addVar()
        y = m.addVar()
        lin = LinExpr([1.0], [y])
        q = (x * x) + lin
        assert isinstance(q, QuadExpr)
        assert q.size() == 1
        assert math.isclose(q.getLinExpr().getCoeff(0), 1.0)

    def test_linexpr_add_quadexpr(self):
        m = fresh_model()
        x = m.addVar()
        y = m.addVar()
        lin = LinExpr([2.0], [y])
        q = lin + (x * x)
        assert isinstance(q, QuadExpr)
        assert q.size() == 1

    def test_quadexpr_add_var(self):
        m = fresh_model()
        x = m.addVar()
        y = m.addVar()
        q = (x * x) + y
        assert isinstance(q, QuadExpr)
        assert q.size() == 1
        lin = q.getLinExpr()
        assert lin.size() == 1

    def test_quadexpr_add_scalar(self):
        m = fresh_model()
        x = m.addVar()
        q = (x * x) + 7.0
        assert isinstance(q, QuadExpr)
        assert math.isclose(q.getLinExpr().constant, 7.0)

    def test_linexpr_times_var(self):
        m = fresh_model()
        x = m.addVar()
        y = m.addVar()
        # (2*x) * y  =>  QuadExpr with term 2.0 * x * y
        lin = 2.0 * x
        q = lin * y
        assert isinstance(q, QuadExpr)
        assert q.size() == 1
        assert math.isclose(q.getCoeff(0), 2.0)
        assert q.getVar1(0) is x
        assert q.getVar2(0) is y

    def test_linexpr_with_constant_times_var(self):
        m = fresh_model()
        x = m.addVar()
        y = m.addVar()
        lin = x + 3.0          # LinExpr: 1*x + 3
        q = lin * y            # QuadExpr: 1*x*y + linear(3*y)
        assert isinstance(q, QuadExpr)
        assert q.size() == 1
        lin_part = q.getLinExpr()
        # constant 3.0 becomes a linear term 3.0*y
        assert lin_part.size() == 1
        assert math.isclose(lin_part.getCoeff(0), 3.0)

    def test_tempqconstr_le(self):
        m = fresh_model()
        x = m.addVar()
        tc = (x * x) <= 4.0
        from grbcompat._quadexpr import TempQConstr
        assert isinstance(tc, TempQConstr)
        assert tc.sense == GRB.LESS_EQUAL

    def test_tempqconstr_ge(self):
        m = fresh_model()
        x = m.addVar()
        tc = (x * x) >= 1.0
        from grbcompat._quadexpr import TempQConstr
        assert isinstance(tc, TempQConstr)
        assert tc.sense == GRB.GREATER_EQUAL

    def test_tempqconstr_eq(self):
        m = fresh_model()
        x = m.addVar()
        tc = (x * x) == 9.0
        from grbcompat._quadexpr import TempQConstr
        assert isinstance(tc, TempQConstr)
        assert tc.sense == GRB.EQUAL

    def test_getlinexpr_returns_copy(self):
        m = fresh_model()
        x = m.addVar()
        y = m.addVar()
        q = (x * x) + y
        lin = q.getLinExpr()
        lin.constant = 999.0  # mutating the copy should not affect q
        assert math.isclose(q.getLinExpr().constant, 0.0)

    def test_radd_zero(self):
        m = fresh_model()
        x = m.addVar()
        q = sum([x * x, x * x])  # sum() starts with 0 + first_item
        assert isinstance(q, QuadExpr)
        assert q.size() == 2

    def test_rsub(self):
        m = fresh_model()
        x = m.addVar()
        q = 5.0 - (x * x)
        assert isinstance(q, QuadExpr)
        assert math.isclose(q.getCoeff(0), -1.0)
        assert math.isclose(q.getLinExpr().constant, 5.0)


# ------------------------------------------------------------------ #
# TestQPObjective – HiGHS solves QP correctly
# ------------------------------------------------------------------ #

class TestQPObjective:

    def test_min_x_squared(self):
        """min x^2, x >= 0  →  x* = 0, obj* = 0"""
        m = fresh_model()
        x = m.addVar(lb=0.0, ub=gp.GRB.INFINITY)
        m.setObjective(x * x, GRB.MINIMIZE)
        m.optimize()
        assert m.Status == GRB.OPTIMAL
        assert math.isclose(x.X, 0.0, abs_tol=1e-6)
        assert math.isclose(m.ObjVal, 0.0, abs_tol=1e-6)

    def test_min_x_squared_with_bound(self):
        """min x^2, x >= 2  →  x* = 2, obj* = 4"""
        m = fresh_model()
        x = m.addVar(lb=2.0)
        m.setObjective(x * x, GRB.MINIMIZE)
        m.optimize()
        assert math.isclose(x.X, 2.0, abs_tol=1e-6)
        assert math.isclose(m.ObjVal, 4.0, abs_tol=1e-6)

    def test_min_diagonal_qp(self):
        """
        min x0^2 + 2*x1^2  s.t. x0 + x1 >= 1, x0,x1 >= 0

        KKT: 2*x0 = λ, 4*x1 = λ, x0 + x1 = 1
        → x0 = 2/3, x1 = 1/3
        """
        m = fresh_model()
        x0 = m.addVar(lb=0.0)
        x1 = m.addVar(lb=0.0)
        m.addConstr(x0 + x1 >= 1.0)
        m.setObjective(x0 * x0 + 2.0 * (x1 * x1), GRB.MINIMIZE)
        m.optimize()
        assert m.Status == GRB.OPTIMAL
        assert math.isclose(x0.X, 2.0 / 3.0, abs_tol=1e-5)
        assert math.isclose(x1.X, 1.0 / 3.0, abs_tol=1e-5)

    def test_min_diagonal_qp_via_plan_example(self):
        """
        min x0^2 + 2*x1^2  s.t. x0 + x1 >= 1  (doc example with x0=5/6, x1=1/6)

        KKT: 2*x0 + 1 = λ, 4*x1 + 2 = λ, x0 + x1 = 1 (with lin terms)
        Wait, this test uses pure QP without linear terms, so x0=2/3, x1=1/3 from above.
        This variant adds linear costs: min x0^2 + x0 + 2*x1^2 + 2*x1
        KKT: 2*x0+1=λ, 4*x1+2=λ → 2*x0-4*x1=1, x0+x1=1
        → x0=1 - x1; 2-2*x1-4*x1=1 → x1=1/6, x0=5/6
        """
        m = fresh_model()
        x0 = m.addVar(lb=0.0)
        x1 = m.addVar(lb=0.0)
        m.addConstr(x0 + x1 >= 1.0)
        obj = x0 * x0 + x0 + 2.0 * (x1 * x1) + 2.0 * x1
        m.setObjective(obj, GRB.MINIMIZE)
        m.optimize()
        assert m.Status == GRB.OPTIMAL
        assert math.isclose(x0.X, 5.0 / 6.0, abs_tol=1e-5)
        assert math.isclose(x1.X, 1.0 / 6.0, abs_tol=1e-5)

    def test_min_cross_term_qp(self):
        """
        min x0^2 + x0*x1 + x1^2 - 3*x0 - 5*x1  (unconstrained, unbounded below?)

        Q = [[2, 1], [1, 2]]; c = [-3, -5]
        Solution: Q*x = -c → [[2,1],[1,2]]*x = [3,5]
        det = 3; x0 = (2*3-5)/3 = 1/3; x1 = (-3+2*5)/3 = 7/3
        """
        m = fresh_model()
        x0 = m.addVar(lb=-gp.GRB.INFINITY)
        x1 = m.addVar(lb=-gp.GRB.INFINITY)
        obj = x0 * x0 + x0 * x1 + x1 * x1 - 3.0 * x0 - 5.0 * x1
        m.setObjective(obj, GRB.MINIMIZE)
        m.optimize()
        assert m.Status == GRB.OPTIMAL
        assert math.isclose(x0.X, 1.0 / 3.0, abs_tol=1e-5)
        assert math.isclose(x1.X, 7.0 / 3.0, abs_tol=1e-5)

    def test_min_linear_and_quad(self):
        """min x^2 - 3*x, x in [0, 2]  →  x* = 1.5"""
        m = fresh_model()
        x = m.addVar(lb=0.0, ub=2.0)
        m.setObjective(x * x - 3.0 * x, GRB.MINIMIZE)
        m.optimize()
        assert m.Status == GRB.OPTIMAL
        assert math.isclose(x.X, 1.5, abs_tol=1e-5)

    def test_reset_from_qp_to_lp(self):
        """setObjective with a LinExpr after QP should clear the Hessian."""
        m = fresh_model()
        x = m.addVar(lb=0.0, ub=1.0)
        y = m.addVar(lb=0.0, ub=1.0)
        # First set a QP objective
        m.setObjective(x * x + y * y, GRB.MINIMIZE)
        assert m._has_quad is True
        # Now reset to a plain linear objective
        m.setObjective(x + y, GRB.MINIMIZE)
        assert m._has_quad is False
        m.optimize()
        assert m.Status == GRB.OPTIMAL
        # LP min x+y, x,y in [0,1] → both at lower bound → obj=0
        assert math.isclose(m.ObjVal, 0.0, abs_tol=1e-6)

    def test_has_quad_flag(self):
        m = fresh_model()
        x = m.addVar()
        assert m._has_quad is False
        m.setObjective(x * x)
        assert m._has_quad is True

    def test_quadexpr_with_constant_offset(self):
        """min x^2 + 10  →  obj* = 10 at x=0"""
        m = fresh_model()
        x = m.addVar(lb=0.0)
        m.setObjective(x * x + 10.0, GRB.MINIMIZE)
        m.optimize()
        assert math.isclose(m.ObjVal, 10.0, abs_tol=1e-5)

    def test_multiple_vars_diagonal(self):
        """min 3*x0^2 + 5*x1^2  s.t. x0>=1, x1>=1  →  obj = 3+5 = 8"""
        m = fresh_model()
        x0 = m.addVar(lb=1.0)
        x1 = m.addVar(lb=1.0)
        m.setObjective(3.0 * (x0 * x0) + 5.0 * (x1 * x1), GRB.MINIMIZE)
        m.optimize()
        assert m.Status == GRB.OPTIMAL
        assert math.isclose(x0.X, 1.0, abs_tol=1e-5)
        assert math.isclose(x1.X, 1.0, abs_tol=1e-5)
        assert math.isclose(m.ObjVal, 8.0, abs_tol=1e-5)

    def test_sense_parameter_in_setObjective(self):
        """setObjective(expr, GRB.MINIMIZE) and GRB.MAXIMIZE respected."""
        m = fresh_model()
        x = m.addVar(lb=0.0, ub=2.0)
        # maximize -x^2, i.e. find maximum of concave function on [0,2]
        # max at x=0 (boundary of concave, but -x^2 is concave so max at x=0 → 0)
        m.setObjective(-(x * x), GRB.MAXIMIZE)
        m.optimize()
        assert m.Status == GRB.OPTIMAL
        assert math.isclose(x.X, 0.0, abs_tol=1e-5)

    def test_getvalue_after_solve(self):
        """QuadExpr.getValue() returns the correct value after optimizing."""
        m = fresh_model()
        x = m.addVar(lb=2.0, ub=2.0)  # fixed at 2
        q = x * x
        m.setObjective(q, GRB.MINIMIZE)
        m.optimize()
        assert math.isclose(q.getValue(), 4.0, abs_tol=1e-5)


# ------------------------------------------------------------------ #
# TestQPCopy – copy() transfers the Hessian
# ------------------------------------------------------------------ #

class TestQPCopy:

    def test_copy_preserves_has_quad(self):
        m = fresh_model()
        x = m.addVar(lb=0.0)
        m.setObjective(x * x, GRB.MINIMIZE)
        c = m.copy()
        assert c._has_quad is True

    def test_copy_lp_has_quad_false(self):
        m = fresh_model()
        x = m.addVar(lb=0.0)
        m.setObjective(x, GRB.MINIMIZE)
        c = m.copy()
        assert c._has_quad is False

    def test_copy_qp_solves_to_same_obj(self):
        """
        min x0^2 + 2*x1^2  s.t. x0+x1>=1
        Copy should solve to the same objective value.
        """
        m = fresh_model()
        x0 = m.addVar(lb=0.0)
        x1 = m.addVar(lb=0.0)
        m.addConstr(x0 + x1 >= 1.0)
        m.setObjective(x0 * x0 + 2.0 * (x1 * x1), GRB.MINIMIZE)
        m.optimize()
        orig_obj = m.ObjVal

        c = m.copy()
        c.optimize()
        assert math.isclose(c.ObjVal, orig_obj, rel_tol=1e-5)

    def test_copy_is_independent(self):
        """Modifying objective on copy does not affect original."""
        m = fresh_model()
        x = m.addVar(lb=1.0)
        m.setObjective(x * x, GRB.MINIMIZE)
        m.optimize()
        orig_obj = m.ObjVal

        c = m.copy()
        cx = c._vars[0]
        c.setObjective(2.0 * (cx * cx), GRB.MINIMIZE)
        c.optimize()

        # original must still give same result
        m.optimize()
        assert math.isclose(m.ObjVal, orig_obj, rel_tol=1e-5)
        # copy should give 2x the original since coefficient doubled and x fixed at 1
        assert math.isclose(c.ObjVal, 2.0 * orig_obj, rel_tol=1e-5)

    def test_copy_qp_solution_values(self):
        """Solution values in copy match original."""
        m = fresh_model()
        x0 = m.addVar(lb=0.0)
        x1 = m.addVar(lb=0.0)
        m.addConstr(x0 + x1 >= 1.0)
        m.setObjective(x0 * x0 + 2.0 * (x1 * x1), GRB.MINIMIZE)
        m.optimize()

        c = m.copy()
        c.optimize()

        cx0, cx1 = c._vars[0], c._vars[1]
        assert math.isclose(cx0.X, x0.X, abs_tol=1e-5)
        assert math.isclose(cx1.X, x1.X, abs_tol=1e-5)


# ------------------------------------------------------------------ #
# TestAddQConstr – stub raises NotImplementedError
# ------------------------------------------------------------------ #

class TestAddQConstr:

    def test_raises_not_implemented(self):
        m = fresh_model()
        x = m.addVar()
        with pytest.raises(NotImplementedError):
            m.addQConstr((x * x) <= 4.0)

    def test_raises_not_implemented_with_name(self):
        m = fresh_model()
        x = m.addVar()
        with pytest.raises(NotImplementedError):
            m.addQConstr((x * x) <= 4.0, name="qc1")


# ------------------------------------------------------------------ #
# TestQPWrite – model.write() round-trips for QP models
# ------------------------------------------------------------------ #

def _make_diagonal_qp():
    """min x0^2 + 2*x1^2 s.t. x0+x1>=1, x0,x1>=0  →  x0=2/3, x1=1/3, obj=2/3"""
    m = fresh_model()
    x0 = m.addVar(lb=0.0, ub=10.0, name="x0")
    x1 = m.addVar(lb=0.0, ub=10.0, name="x1")
    m.addConstr(x0 + x1 >= 1.0, name="cap")
    m.setObjective(x0 * x0 + 2.0 * (x1 * x1), GRB.MINIMIZE)
    return m, x0, x1


def _make_cross_term_qp():
    """min x0^2 + x0*x1 + x1^2 - 3*x0 - 5*x1  →  x0=1/3, x1=7/3, obj=-34/3"""
    m = fresh_model()
    x0 = m.addVar(lb=-gp.GRB.INFINITY, name="x0")
    x1 = m.addVar(lb=-gp.GRB.INFINITY, name="x1")
    m.setObjective(x0 * x0 + x0 * x1 + x1 * x1 - 3.0 * x0 - 5.0 * x1, GRB.MINIMIZE)
    return m, x0, x1


class TestQPWrite:

    # ---------------------------------------------------------------- #
    # LP format
    # ---------------------------------------------------------------- #

    def test_write_lp_creates_file(self):
        m, _, _ = _make_diagonal_qp()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "qp.lp")
            m.write(path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0

    def test_write_lp_contains_quadratic_section(self):
        m, _, _ = _make_diagonal_qp()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "qp.lp")
            m.write(path)
            content = open(path).read()
            # HiGHS LP format uses [ ... ]/2 notation for the quadratic part
            assert "[" in content and "]" in content

    def test_write_lp_roundtrip_solves_correctly(self):
        m, x0, x1 = _make_diagonal_qp()
        m.optimize()
        orig_obj = m.ObjVal
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "qp.lp")
            m.write(path)
            m2 = fresh_model()
            m2.read(path)
            m2.optimize()
            assert m2.Status == GRB.OPTIMAL
            assert math.isclose(m2.ObjVal, orig_obj, rel_tol=1e-5)

    def test_write_lp_roundtrip_solution_values(self):
        m, x0, x1 = _make_diagonal_qp()
        m.optimize()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "qp.lp")
            m.write(path)
            m2 = fresh_model()
            m2.read(path)
            m2.optimize()
            v0, v1 = m2._vars[0], m2._vars[1]
            assert math.isclose(v0.X, x0.X, abs_tol=1e-5)
            assert math.isclose(v1.X, x1.X, abs_tol=1e-5)

    def test_write_lp_sets_has_quad_on_read(self):
        m, _, _ = _make_diagonal_qp()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "qp.lp")
            m.write(path)
            m2 = fresh_model()
            assert m2._has_quad is False
            m2.read(path)
            assert m2._has_quad is True

    # ---------------------------------------------------------------- #
    # MPS format
    # ---------------------------------------------------------------- #

    def test_write_mps_creates_file(self):
        m, _, _ = _make_diagonal_qp()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "qp.mps")
            m.write(path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0

    def test_write_mps_contains_quadobj_section(self):
        m, _, _ = _make_diagonal_qp()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "qp.mps")
            m.write(path)
            content = open(path).read()
            assert "QUADOBJ" in content

    def test_write_mps_roundtrip_solves_correctly(self):
        m, _, _ = _make_diagonal_qp()
        m.optimize()
        orig_obj = m.ObjVal
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "qp.mps")
            m.write(path)
            m2 = fresh_model()
            m2.read(path)
            m2.optimize()
            assert m2.Status == GRB.OPTIMAL
            assert math.isclose(m2.ObjVal, orig_obj, rel_tol=1e-5)

    def test_write_mps_sets_has_quad_on_read(self):
        m, _, _ = _make_diagonal_qp()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "qp.mps")
            m.write(path)
            m2 = fresh_model()
            m2.read(path)
            assert m2._has_quad is True

    # ---------------------------------------------------------------- #
    # Cross-term QP
    # ---------------------------------------------------------------- #

    def test_cross_term_lp_roundtrip(self):
        m, x0, x1 = _make_cross_term_qp()
        m.optimize()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "cross.lp")
            m.write(path)
            m2 = fresh_model()
            m2.read(path)
            m2.optimize()
            assert math.isclose(m2.ObjVal, m.ObjVal, rel_tol=1e-5)
            assert math.isclose(m2._vars[0].X, x0.X, abs_tol=1e-5)
            assert math.isclose(m2._vars[1].X, x1.X, abs_tol=1e-5)

    def test_cross_term_mps_roundtrip(self):
        m, x0, x1 = _make_cross_term_qp()
        m.optimize()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "cross.mps")
            m.write(path)
            m2 = fresh_model()
            m2.read(path)
            m2.optimize()
            assert math.isclose(m2.ObjVal, m.ObjVal, rel_tol=1e-5)

    # ---------------------------------------------------------------- #
    # LP-only write (no Hessian) preserves LP semantics
    # ---------------------------------------------------------------- #

    def test_write_lp_after_reset_to_linear(self):
        """setObjective with LinExpr after QP → write → read → _has_quad is False."""
        m = fresh_model()
        x = m.addVar(lb=0.0, ub=5.0)
        m.setObjective(x * x, GRB.MINIMIZE)
        assert m._has_quad is True
        m.setObjective(x, GRB.MINIMIZE)       # reset to LP
        assert m._has_quad is False
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "lp_after_reset.lp")
            m.write(path)
            m2 = fresh_model()
            m2.read(path)
            assert m2._has_quad is False
            m2.optimize()
            assert math.isclose(m2._vars[0].X, 0.0, abs_tol=1e-5)

    # ---------------------------------------------------------------- #
    # Unsolved model write (no crash before optimize)
    # ---------------------------------------------------------------- #

    def test_write_qp_before_optimize(self):
        """Writing a QP model that has never been optimised should not raise."""
        m, _, _ = _make_diagonal_qp()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "unsolved.lp")
            m.write(path)   # must not raise
            assert os.path.getsize(path) > 0
