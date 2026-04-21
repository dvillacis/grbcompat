"""Tests for Model.read() — loading LP/MPS files into the wrapper."""

import os
import tempfile

import pytest
from grbcompat import GRB, Model
from grbcompat._constr import Constr
from grbcompat._var import Var


TOL = 1e-6


# ------------------------------------------------------------------ #
# Fixtures / helpers
# ------------------------------------------------------------------ #

def _write_then_read(model: Model, suffix: str) -> Model:
    """Write *model* to a temp file and read it into a fresh Model."""
    path = tempfile.mktemp(suffix=suffix)
    try:
        model.write(path)
        m2 = Model()
        m2.read(path)
    finally:
        if os.path.exists(path):
            os.unlink(path)
    return m2


def _build_lp():
    """
    min  2x + 3y
    s.t. x + y >= 4   (GE)
         x     <= 6   (LE)
         x, y >= 0
    Optimal: x=4, y=0, obj=8.
    """
    m = Model()
    m.addVar(lb=0.0, ub=10.0, name="x")
    m.addVar(lb=0.0, name="y")
    m.addConstr(m.getVars()[0] + m.getVars()[1] >= 4)
    m.addConstr(m.getVars()[0] <= 6)
    m.setObjective(2 * m.getVars()[0] + 3 * m.getVars()[1], GRB.MINIMIZE)
    return m


def _build_lp_named():
    """Same LP but with explicit user names (lost after write/read)."""
    m = Model()
    x = m.addVar(lb=0.0, ub=10.0, name="x")
    y = m.addVar(lb=0.0, name="y")
    m.addConstr(x + y >= 4, name="demand")
    m.addConstr(x <= 6, name="cap")
    m.addConstr(x + y == 5, name="eq_c")
    m.setObjective(2 * x + 3 * y, GRB.MINIMIZE)
    return m


def _build_mip():
    """
    max  2*b0 + b1 + z
    s.t. b0 + b1 <= 1
         z <= 3
         b0, b1 binary; z integer >= 0
    Optimal: b0=1, b1=0, z=3, obj=5.
    """
    m = Model()
    b0 = m.addVar(vtype=GRB.BINARY, name="b0")
    b1 = m.addVar(vtype=GRB.BINARY, name="b1")
    z  = m.addVar(lb=0.0, vtype=GRB.INTEGER, name="z")
    m.addConstr(b0 + b1 <= 1)
    m.addConstr(z <= 3)
    m.setObjective(2 * b0 + b1 + z, GRB.MAXIMIZE)
    return m


# ------------------------------------------------------------------ #
# Basic structure
# ------------------------------------------------------------------ #

class TestReadBasicStructure:
    @pytest.mark.parametrize("suffix", [".lp", ".mps"])
    def test_num_vars(self, suffix):
        cp = _write_then_read(_build_lp(), suffix)
        assert cp.NumVars == 2

    @pytest.mark.parametrize("suffix", [".lp", ".mps"])
    def test_num_constrs(self, suffix):
        cp = _write_then_read(_build_lp(), suffix)
        assert cp.NumConstrs == 2

    def test_getvars_returns_var_objects(self):
        cp = _write_then_read(_build_lp(), ".lp")
        assert all(isinstance(v, Var) for v in cp.getVars())

    def test_getconstrs_returns_constr_objects(self):
        cp = _write_then_read(_build_lp(), ".lp")
        assert all(isinstance(c, Constr) for c in cp.getConstrs())

    def test_status_is_loaded(self):
        cp = _write_then_read(_build_lp(), ".lp")
        assert cp.Status == GRB.LOADED

    def test_read_into_existing_model_replaces_state(self):
        """Reading into a model that already has vars clears the old state."""
        m_old = Model()
        m_old.addVar(name="old")
        assert m_old.NumVars == 1

        path = tempfile.mktemp(suffix=".lp")
        try:
            _build_lp().write(path)
            m_old.read(path)
        finally:
            if os.path.exists(path):
                os.unlink(path)

        assert m_old.NumVars == 2
        # Old var "old" is gone; new names are HiGHS-generated
        assert m_old.getVarByName("old") is None

    def test_col_indices_are_sequential(self):
        cp = _write_then_read(_build_lp(), ".lp")
        for i, v in enumerate(cp.getVars()):
            assert v._col_idx == i

    def test_row_indices_are_sequential(self):
        cp = _write_then_read(_build_lp(), ".lp")
        for i, c in enumerate(cp.getConstrs()):
            assert c._row_idx == i


# ------------------------------------------------------------------ #
# Variable metadata
# ------------------------------------------------------------------ #

class TestReadVarMetadata:
    def setup_method(self):
        self.cp = _write_then_read(_build_lp_named(), ".lp")
        self.vars = self.cp.getVars()  # c0=x, c1=y

    def test_var_lb_first(self):
        assert self.vars[0].LB == pytest.approx(0.0)

    def test_var_ub_first(self):
        assert self.vars[0].UB == pytest.approx(10.0)

    def test_var_lb_second(self):
        assert self.vars[1].LB == pytest.approx(0.0)

    def test_var_ub_second_is_infinity(self):
        import math
        assert math.isinf(self.vars[1].UB)

    def test_var_names_are_highs_generated(self):
        # HiGHS auto-generates names c0, c1, ... when user names aren't
        # registered in HiGHS (the wrapper stores them only in Python).
        names = [v.VarName for v in self.vars]
        assert names[0] == "c0"
        assert names[1] == "c1"

    def test_getvarbyname_works(self):
        assert self.cp.getVarByName("c0") is self.vars[0]

    def test_continuous_vtype(self):
        cp = _write_then_read(_build_lp(), ".lp")
        for v in cp.getVars():
            assert v.VType == GRB.CONTINUOUS

    def test_integer_vtype(self):
        cp = _write_then_read(_build_mip(), ".lp")
        vtypes = [v.VType for v in cp.getVars()]
        assert GRB.INTEGER in vtypes

    def test_binary_vtype(self):
        cp = _write_then_read(_build_mip(), ".lp")
        vtypes = [v.VType for v in cp.getVars()]
        assert GRB.BINARY in vtypes

    def test_binary_bounds_after_read(self):
        cp = _write_then_read(_build_mip(), ".lp")
        binary_vars = [v for v in cp.getVars() if v.VType == GRB.BINARY]
        for v in binary_vars:
            assert v.LB == pytest.approx(0.0)
            assert v.UB == pytest.approx(1.0)


# ------------------------------------------------------------------ #
# Constraint metadata
# ------------------------------------------------------------------ #

class TestReadConstrMetadata:
    def setup_method(self):
        self.cp = _write_then_read(_build_lp_named(), ".lp")
        self.constrs = self.cp.getConstrs()  # r0=demand(GE), r1=cap(LE), r2=eq

    def test_constr_sense_ge(self):
        assert self.constrs[0].Sense == GRB.GREATER_EQUAL

    def test_constr_sense_le(self):
        assert self.constrs[1].Sense == GRB.LESS_EQUAL

    def test_constr_sense_eq(self):
        assert self.constrs[2].Sense == GRB.EQUAL

    def test_constr_rhs_ge(self):
        assert self.constrs[0].RHS == pytest.approx(4.0)

    def test_constr_rhs_le(self):
        assert self.constrs[1].RHS == pytest.approx(6.0)

    def test_constr_rhs_eq(self):
        assert self.constrs[2].RHS == pytest.approx(5.0)

    def test_constr_names_are_highs_generated(self):
        names = [c.ConstrName for c in self.constrs]
        assert names[0] == "r0"
        assert names[1] == "r1"

    def test_getconstrbyname_works(self):
        assert self.cp.getConstrByName("r0") is self.constrs[0]


# ------------------------------------------------------------------ #
# Objective
# ------------------------------------------------------------------ #

class TestReadObjective:
    def test_minimize_sense_lp(self):
        cp = _write_then_read(_build_lp(), ".lp")
        assert cp.ModelSense == GRB.MINIMIZE

    def test_minimize_sense_mps(self):
        cp = _write_then_read(_build_lp(), ".mps")
        assert cp.ModelSense == GRB.MINIMIZE

    def test_maximize_sense_lp(self):
        m = Model()
        x = m.addVar(lb=0.0, ub=5.0)
        m.setObjective(x, GRB.MAXIMIZE)
        cp = _write_then_read(m, ".lp")
        assert cp.ModelSense == GRB.MAXIMIZE

    def test_maximize_sense_mps(self):
        m = Model()
        x = m.addVar(lb=0.0, ub=5.0)
        m.setObjective(x, GRB.MAXIMIZE)
        cp = _write_then_read(m, ".mps")
        assert cp.ModelSense == GRB.MAXIMIZE


# ------------------------------------------------------------------ #
# Solve after read
# ------------------------------------------------------------------ #

class TestReadAndSolve:
    @pytest.mark.parametrize("suffix", [".lp", ".mps"])
    def test_roundtrip_optimal_status(self, suffix):
        cp = _write_then_read(_build_lp(), suffix)
        cp.optimize()
        assert cp.Status == GRB.OPTIMAL

    @pytest.mark.parametrize("suffix", [".lp", ".mps"])
    def test_roundtrip_same_obj(self, suffix):
        m = _build_lp()
        m.optimize()
        cp = _write_then_read(m, suffix)
        cp.optimize()
        assert cp.ObjVal == pytest.approx(m.ObjVal, abs=TOL)

    def test_primal_values_correct_after_read(self):
        cp = _write_then_read(_build_lp(), ".lp")
        cp.optimize()
        x, y = cp.getVars()
        assert x.X == pytest.approx(4.0, abs=TOL)
        assert y.X == pytest.approx(0.0, abs=TOL)

    def test_duals_accessible_after_read_and_solve(self):
        cp = _write_then_read(_build_lp(), ".lp")
        cp.optimize()
        for c in cp.getConstrs():
            assert isinstance(c.Pi, float)

    def test_mip_roundtrip_same_obj(self):
        m = _build_mip()
        m.optimize()
        cp = _write_then_read(m, ".lp")
        cp.optimize()
        assert cp.ObjVal == pytest.approx(m.ObjVal, abs=TOL)

    def test_read_and_modify_bound_then_solve(self):
        """After read, modifying a bound and re-solving gives the new optimum."""
        cp = _write_then_read(_build_lp(), ".lp")
        # Tighten the UB on x from 10 to 2.
        cp.getVars()[0].UB = 2.0
        cp.optimize()
        assert cp.Status == GRB.OPTIMAL
        # With x<=2 and x+y>=4: y>=2, so obj = 2*2 + 3*2 = 10
        assert cp.ObjVal == pytest.approx(10.0, abs=TOL)

    def test_read_and_add_constr_then_solve(self):
        """After read, adding a constraint and re-solving works."""
        cp = _write_then_read(_build_lp(), ".lp")
        x = cp.getVars()[0]
        cp.addConstr(x >= 5)  # force x >= 5
        cp.optimize()
        assert cp.Status == GRB.OPTIMAL
        assert x.X == pytest.approx(5.0, abs=TOL)

    def test_read_then_remove_constr_and_solve(self):
        """After read, removing a constraint and re-solving works."""
        cp = _write_then_read(_build_lp(), ".lp")
        # Remove the GE constraint (r0: x+y>=4); only x<=6 remains
        ge_constr = cp.getConstrs()[0]
        cp.remove(ge_constr)
        cp.optimize()
        assert cp.Status == GRB.OPTIMAL
        # Without demand constraint, min 2x+3y with x,y>=0 → x=y=0, obj=0
        assert cp.ObjVal == pytest.approx(0.0, abs=TOL)

    def test_read_then_copy_and_solve(self):
        """A model loaded from file can be copied and both solve correctly."""
        cp = _write_then_read(_build_lp(), ".lp")
        cp2 = cp.copy()
        cp.optimize()
        cp2.optimize()
        assert cp.ObjVal == pytest.approx(cp2.ObjVal, abs=TOL)
