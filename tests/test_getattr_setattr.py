"""Tests for Model.getAttr / Model.setAttr completeness."""

import math
import pytest
from grbcompat import GRB, Model


TOL = 1e-6


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _build_lp():
    """
    min 2x + 3y  s.t.  x + y >= 4, x <= 6,  x,y >= 0
    Optimal: x=4, y=0, obj=8.
    """
    m = Model()
    x = m.addVar(lb=0.0, ub=10.0, obj=0.0, name="x")
    y = m.addVar(lb=0.0, ub=GRB.INFINITY, obj=0.0, name="y")
    c0 = m.addConstr(x + y >= 4, name="dem")
    c1 = m.addConstr(x <= 6, name="cap")
    m.setObjective(2 * x + 3 * y, GRB.MINIMIZE)
    return m, x, y, c0, c1


def _build_mip():
    """
    max  b0 + 2*b1  s.t.  b0 + b1 <= 1
    Optimal: b1=1, obj=2.
    """
    m = Model()
    b0 = m.addVar(vtype=GRB.BINARY, name="b0")
    b1 = m.addVar(vtype=GRB.BINARY, name="b1")
    m.addConstr(b0 + b1 <= 1)
    m.setObjective(b0 + 2 * b1, GRB.MAXIMIZE)
    return m, b0, b1


# ================================================================== #
# getAttr — variable attributes (batch)
# ================================================================== #

class TestGetAttrVars:
    def test_x_after_solve(self):
        m, x, y, *_ = _build_lp()
        m.optimize()
        vals = m.getAttr("X", [x, y])
        assert vals[0] == pytest.approx(4.0, abs=TOL)
        assert vals[1] == pytest.approx(0.0, abs=TOL)

    def test_rc_after_solve(self):
        m, x, y, *_ = _build_lp()
        m.optimize()
        vals = m.getAttr("RC", [x, y])
        assert all(isinstance(v, float) for v in vals)

    def test_lb(self):
        m, x, y, *_ = _build_lp()
        vals = m.getAttr("LB", [x, y])
        assert vals[0] == pytest.approx(0.0)
        assert vals[1] == pytest.approx(0.0)

    def test_ub(self):
        m, x, y, *_ = _build_lp()
        vals = m.getAttr("UB", [x, y])
        assert vals[0] == pytest.approx(10.0)
        assert math.isinf(vals[1])

    def test_obj(self):
        m, x, y, *_ = _build_lp()
        # addVar default obj=0; setObjective updates HiGHS costs but NOT _var_objs
        vals = m.getAttr("Obj", [x, y])
        assert vals == [pytest.approx(0.0), pytest.approx(0.0)]

    def test_varname(self):
        m, x, y, *_ = _build_lp()
        vals = m.getAttr("VarName", [x, y])
        assert vals == ["x", "y"]

    def test_vtype_continuous(self):
        m, x, y, *_ = _build_lp()
        vals = m.getAttr("VType", [x, y])
        assert vals == [GRB.CONTINUOUS, GRB.CONTINUOUS]

    def test_vtype_binary(self):
        m, b0, b1 = _build_mip()
        vals = m.getAttr("VType", [b0, b1])
        assert vals == [GRB.BINARY, GRB.BINARY]

    def test_start_default_is_undefined(self):
        m, x, y, *_ = _build_lp()
        vals = m.getAttr("Start", [x, y])
        assert all(v >= GRB.INFINITY for v in vals)

    def test_start_after_set(self):
        m, x, y, *_ = _build_lp()
        x.Start = 3.0
        vals = m.getAttr("Start", [x, y])
        assert vals[0] == pytest.approx(3.0)
        assert vals[1] >= GRB.INFINITY  # still UNDEFINED

    def test_branchpriority_default_is_zero(self):
        m, x, y, *_ = _build_lp()
        vals = m.getAttr("BranchPriority", [x, y])
        assert vals == [0, 0]

    def test_branchpriority_after_set(self):
        m, x, y, *_ = _build_lp()
        x.BranchPriority = 5
        vals = m.getAttr("BranchPriority", [x, y])
        assert vals[0] == 5
        assert vals[1] == 0

    def test_unknown_attr_raises(self):
        m, x, y, *_ = _build_lp()
        with pytest.raises(AttributeError):
            m.getAttr("Nonexistent", [x, y])


# ================================================================== #
# getAttr — constraint attributes (batch)
# ================================================================== #

class TestGetAttrConstrs:
    def test_pi_after_solve(self):
        m, x, y, c0, c1 = _build_lp()
        m.optimize()
        vals = m.getAttr("Pi", [c0, c1])
        assert all(isinstance(v, float) for v in vals)

    def test_slack_after_solve(self):
        m, x, y, c0, c1 = _build_lp()
        m.optimize()
        vals = m.getAttr("Slack", [c0, c1])
        assert all(isinstance(v, float) for v in vals)

    def test_rhs(self):
        m, x, y, c0, c1 = _build_lp()
        vals = m.getAttr("RHS", [c0, c1])
        assert vals[0] == pytest.approx(4.0)
        assert vals[1] == pytest.approx(6.0)

    def test_sense(self):
        m, x, y, c0, c1 = _build_lp()
        vals = m.getAttr("Sense", [c0, c1])
        assert vals[0] == GRB.GREATER_EQUAL
        assert vals[1] == GRB.LESS_EQUAL

    def test_constrname(self):
        m, x, y, c0, c1 = _build_lp()
        vals = m.getAttr("ConstrName", [c0, c1])
        assert vals == ["dem", "cap"]


# ================================================================== #
# getAttr — model-level attributes (no objs)
# ================================================================== #

class TestGetAttrModel:
    def test_objval_after_solve(self):
        m, *_ = _build_lp()
        m.optimize()
        assert m.getAttr("ObjVal") == pytest.approx(8.0, abs=TOL)

    def test_numvars(self):
        m, *_ = _build_lp()
        assert m.getAttr("NumVars") == 2

    def test_numconstrs(self):
        m, *_ = _build_lp()
        assert m.getAttr("NumConstrs") == 2

    def test_modelsense_minimize(self):
        m, *_ = _build_lp()
        assert m.getAttr("ModelSense") == GRB.MINIMIZE

    def test_modelsense_maximize(self):
        m, b0, b1 = _build_mip()
        assert m.getAttr("ModelSense") == GRB.MAXIMIZE

    def test_objcon_default_zero(self):
        m, *_ = _build_lp()
        assert m.getAttr("ObjCon") == pytest.approx(0.0)

    def test_objcon_after_setobjective_with_constant(self):
        m = Model()
        x = m.addVar(lb=2.0, ub=2.0)
        m.setObjective(x + 10, GRB.MINIMIZE)
        assert m.getAttr("ObjCon") == pytest.approx(10.0)

    def test_objcon_after_setattr(self):
        m, *_ = _build_lp()
        m.setAttr("ObjCon", 7.0)
        assert m.getAttr("ObjCon") == pytest.approx(7.0)

    def test_objcon_property_matches_getattr(self):
        m = Model()
        x = m.addVar(lb=1.0, ub=1.0)
        m.setObjective(x + 5, GRB.MINIMIZE)
        assert m.ObjCon == pytest.approx(m.getAttr("ObjCon"))


# ================================================================== #
# setAttr — batch variable attributes
# ================================================================== #

class TestSetAttrVars:
    def test_set_lb(self):
        m, x, y, c0, c1 = _build_lp()
        m.setAttr("LB", [x, y], [1.0, 2.0])
        assert x.LB == pytest.approx(1.0)
        assert y.LB == pytest.approx(2.0)

    def test_set_ub(self):
        m, x, y, *_ = _build_lp()
        m.setAttr("UB", [x, y], [3.0, 5.0])
        assert x.UB == pytest.approx(3.0)
        assert y.UB == pytest.approx(5.0)

    def test_set_obj(self):
        m, x, y, *_ = _build_lp()
        m.setAttr("Obj", [x, y], [4.0, 5.0])
        assert x.Obj == pytest.approx(4.0)
        assert y.Obj == pytest.approx(5.0)

    def test_set_vtype(self):
        m = Model()
        x = m.addVar(lb=0.0, ub=1.0, name="x")
        y = m.addVar(lb=0.0, ub=1.0, name="y")
        m.setAttr("VType", [x, y], [GRB.BINARY, GRB.INTEGER])
        assert x.VType == GRB.BINARY
        assert y.VType == GRB.INTEGER

    def test_set_varname(self):
        m, x, y, *_ = _build_lp()
        m.setAttr("VarName", [x, y], ["alpha", "beta"])
        assert x.VarName == "alpha"
        assert y.VarName == "beta"

    def test_set_start(self):
        m, x, y, *_ = _build_lp()
        m.setAttr("Start", [x, y], [3.5, 0.5])
        assert x.Start == pytest.approx(3.5)
        assert y.Start == pytest.approx(0.5)

    def test_set_branchpriority(self):
        m, x, y, *_ = _build_lp()
        m.setAttr("BranchPriority", [x, y], [10, 20])
        assert x.BranchPriority == 10
        assert y.BranchPriority == 20

    def test_unknown_batch_attr_raises(self):
        m, x, y, *_ = _build_lp()
        with pytest.raises(AttributeError):
            m.setAttr("Nonexistent", [x], [1.0])

    def test_set_lb_affects_solve(self):
        """Setting LB via setAttr changes the optimal solution."""
        m, x, y, c0, c1 = _build_lp()
        m.setAttr("LB", [x], [5.0])   # force x >= 5
        m.optimize()
        assert m.Status == GRB.OPTIMAL
        assert x.X >= 5.0 - TOL

    def test_set_ub_affects_solve(self):
        """Setting UB via setAttr changes the optimal solution."""
        m, x, y, c0, c1 = _build_lp()
        m.setAttr("UB", [x], [2.0])   # x <= 2, need y >= 2 for demand
        m.optimize()
        assert m.Status == GRB.OPTIMAL
        assert m.ObjVal == pytest.approx(10.0, abs=TOL)


# ================================================================== #
# setAttr — batch constraint attributes
# ================================================================== #

class TestSetAttrConstrs:
    def test_set_rhs(self):
        m, x, y, c0, c1 = _build_lp()
        m.setAttr("RHS", [c0, c1], [6.0, 8.0])
        assert c0.RHS == pytest.approx(6.0)
        assert c1.RHS == pytest.approx(8.0)

    def test_set_rhs_affects_solve(self):
        m, x, y, c0, c1 = _build_lp()
        m.setAttr("RHS", [c0], [2.0])   # relax demand to x+y >= 2
        m.optimize()
        assert m.Status == GRB.OPTIMAL
        assert m.ObjVal == pytest.approx(4.0, abs=TOL)  # x=2,y=0 → 2*2=4

    def test_set_constrname(self):
        m, x, y, c0, c1 = _build_lp()
        m.setAttr("ConstrName", [c0, c1], ["demand2", "cap2"])
        assert c0.ConstrName == "demand2"
        assert c1.ConstrName == "cap2"

    def test_set_sense(self):
        """Change sense from >= to <= and verify it's stored."""
        m, x, y, c0, c1 = _build_lp()
        m.setAttr("Sense", [c0], [GRB.LESS_EQUAL])
        assert c0.Sense == GRB.LESS_EQUAL

    def test_set_sense_affects_solve(self):
        """
        Original: x + y >= 4 (GE), x <= 6.
        Change c0 to x + y <= 2 (LE, RHS unchanged but sense flipped).
        Then min 2x+3y with x+y<=4 and x<=6: optimal x=y=0, obj=0.
        """
        m, x, y, c0, c1 = _build_lp()
        # Flip sense: was x+y >= 4, becomes x+y <= 4
        m.setAttr("Sense", [c0], [GRB.LESS_EQUAL])
        m.optimize()
        assert m.Status == GRB.OPTIMAL
        assert m.ObjVal == pytest.approx(0.0, abs=TOL)


# ================================================================== #
# setAttr — model-level
# ================================================================== #

class TestSetAttrModel:
    def test_set_modelsense(self):
        m, *_ = _build_lp()
        m.setAttr("ModelSense", GRB.MAXIMIZE)
        assert m.ModelSense == GRB.MAXIMIZE

    def test_set_objcon(self):
        m, *_ = _build_lp()
        m.setAttr("ObjCon", 100.0)
        assert m.ObjCon == pytest.approx(100.0)

    def test_set_objcon_affects_objval(self):
        m = Model()
        x = m.addVar(lb=1.0, ub=1.0)
        m.setObjective(x, GRB.MINIMIZE)
        m.setAttr("ObjCon", 5.0)
        m.optimize()
        assert m.ObjVal == pytest.approx(6.0, abs=TOL)  # 1 + 5

    def test_objcon_property_setter(self):
        m, *_ = _build_lp()
        m.ObjCon = 3.0
        assert m.getAttr("ObjCon") == pytest.approx(3.0)


# ================================================================== #
# Var.Start attribute
# ================================================================== #

class TestVarStart:
    def test_default_is_undefined(self):
        m = Model()
        x = m.addVar(lb=0.0, ub=10.0)
        assert x.Start >= GRB.INFINITY

    def test_set_and_get(self):
        m = Model()
        x = m.addVar(lb=0.0, ub=10.0)
        x.Start = 4.5
        assert x.Start == pytest.approx(4.5)

    def test_set_to_undefined(self):
        m = Model()
        x = m.addVar(lb=0.0, ub=10.0)
        x.Start = 4.5
        x.Start = GRB.UNDEFINED
        assert x.Start >= GRB.INFINITY

    def test_start_applied_before_optimize(self):
        """Setting a valid start hint doesn't prevent finding the optimum."""
        m, b0, b1 = _build_mip()
        b0.Start = 1.0
        b1.Start = 0.0
        m.optimize()
        assert m.Status == GRB.OPTIMAL
        assert m.ObjVal == pytest.approx(2.0, abs=TOL)  # b1=1 is optimal

    def test_removed_var_start_raises(self):
        m = Model()
        x = m.addVar(lb=0.0, ub=5.0)
        m.remove(x)
        with pytest.raises(RuntimeError):
            _ = x.Start

    def test_copy_preserves_starts(self):
        m, x, y, *_ = _build_lp()
        x.Start = 3.0
        cp = m.copy()
        assert cp.getVars()[0].Start == pytest.approx(3.0)
        assert cp.getVars()[1].Start >= GRB.INFINITY


# ================================================================== #
# Var.BranchPriority attribute
# ================================================================== #

class TestVarBranchPriority:
    def test_default_is_zero(self):
        m = Model()
        x = m.addVar(lb=0.0, ub=5.0)
        assert x.BranchPriority == 0

    def test_set_and_get(self):
        m = Model()
        x = m.addVar(lb=0.0, ub=5.0)
        x.BranchPriority = 10
        assert x.BranchPriority == 10

    def test_stored_as_int(self):
        m = Model()
        x = m.addVar(lb=0.0, ub=5.0)
        x.BranchPriority = 3.9
        assert isinstance(x.BranchPriority, int)
        assert x.BranchPriority == 3

    def test_removed_var_priority_raises(self):
        m = Model()
        x = m.addVar(lb=0.0, ub=5.0)
        m.remove(x)
        with pytest.raises(RuntimeError):
            _ = x.BranchPriority

    def test_copy_preserves_priorities(self):
        m, x, y, *_ = _build_lp()
        x.BranchPriority = 7
        cp = m.copy()
        assert cp.getVars()[0].BranchPriority == 7
        assert cp.getVars()[1].BranchPriority == 0


# ================================================================== #
# Constr.Sense setter
# ================================================================== #

class TestConstrSenseSetter:
    def test_set_ge_to_le(self):
        m, x, y, c0, c1 = _build_lp()
        assert c0.Sense == GRB.GREATER_EQUAL
        c0.Sense = GRB.LESS_EQUAL
        assert c0.Sense == GRB.LESS_EQUAL

    def test_set_le_to_ge(self):
        m, x, y, c0, c1 = _build_lp()
        c1.Sense = GRB.GREATER_EQUAL
        assert c1.Sense == GRB.GREATER_EQUAL

    def test_set_to_equal(self):
        m, x, y, c0, c1 = _build_lp()
        c0.Sense = GRB.EQUAL
        assert c0.Sense == GRB.EQUAL

    def test_sense_change_affects_solve(self):
        """
        Original: x + y >= 4 (GE), x <= 6, min 2x+3y.
        Change c0 to <= 4: demand is now an upper bound, min hits 0.
        """
        m, x, y, c0, c1 = _build_lp()
        c0.Sense = GRB.LESS_EQUAL
        m.optimize()
        assert m.Status == GRB.OPTIMAL
        assert m.ObjVal == pytest.approx(0.0, abs=TOL)

    def test_removed_constr_sense_setter_raises(self):
        m, x, y, c0, c1 = _build_lp()
        m.remove(c0)
        with pytest.raises(RuntimeError):
            c0.Sense = GRB.LESS_EQUAL


# ================================================================== #
# ObjCon property on Model
# ================================================================== #

class TestObjConProperty:
    def test_default_zero(self):
        m = Model()
        assert m.ObjCon == pytest.approx(0.0)

    def test_after_setobjective_constant(self):
        m = Model()
        x = m.addVar(lb=0.0, ub=5.0)
        m.setObjective(x + 7, GRB.MINIMIZE)
        assert m.ObjCon == pytest.approx(7.0)

    def test_after_setattr(self):
        m = Model()
        m.setAttr("ObjCon", -3.5)
        assert m.ObjCon == pytest.approx(-3.5)

    def test_property_setter(self):
        m = Model()
        m.ObjCon = 12.0
        assert m.ObjCon == pytest.approx(12.0)

    def test_affects_objval(self):
        m = Model()
        x = m.addVar(lb=2.0, ub=2.0)
        m.setObjective(x, GRB.MINIMIZE)
        m.ObjCon = 8.0
        m.optimize()
        assert m.ObjVal == pytest.approx(10.0, abs=TOL)

    def test_preserved_in_copy(self):
        m = Model()
        x = m.addVar(lb=0.0, ub=1.0)
        m.setObjective(x + 4, GRB.MINIMIZE)
        cp = m.copy()
        assert cp.ObjCon == pytest.approx(4.0)

    def test_preserved_after_read(self):
        """ObjCon round-trips through write/read."""
        import os
        import tempfile
        m = Model()
        x = m.addVar(lb=1.0, ub=1.0)
        m.setObjective(x + 3, GRB.MINIMIZE)
        path = tempfile.mktemp(suffix=".lp")
        try:
            m.write(path)
            m2 = Model()
            m2.read(path)
            m2.optimize()
            assert m2.ObjVal == pytest.approx(4.0, abs=TOL)
        finally:
            if os.path.exists(path):
                os.unlink(path)
