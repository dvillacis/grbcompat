"""Tests for the Var class."""

import pytest
from grbcompat import GRB, Model
from grbcompat._expr import LinExpr, TempConstr


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def solved_model():
    """
    minimize x + y  s.t.  x + y >= 2, x <= 5, x,y >= 0
    Optimal: x=2, y=0.
    """
    m = Model()
    x = m.addVar(lb=0.0, name="x")
    y = m.addVar(lb=0.0, name="y")
    m.addConstr(x + y >= 2)
    m.setObjective(x + y, GRB.MINIMIZE)
    m.optimize()
    return m, x, y


# ------------------------------------------------------------------ #
# Construction and metadata
# ------------------------------------------------------------------ #

class TestVarConstruction:
    def test_var_has_correct_col_index(self):
        m = Model()
        x = m.addVar(name="x")
        y = m.addVar(name="y")
        assert x._col_idx == 0
        assert y._col_idx == 1

    def test_var_name(self):
        m = Model()
        x = m.addVar(name="my_var")
        assert x.VarName == "my_var"

    def test_varname_setter(self):
        m = Model()
        x = m.addVar(name="original")
        x.VarName = "renamed"
        assert x.VarName == "renamed"

    def test_lb_default(self):
        m = Model()
        x = m.addVar()
        assert x.LB == pytest.approx(0.0)

    def test_ub_default(self):
        m = Model()
        x = m.addVar()
        assert x.UB == float("inf")

    def test_lb_custom(self):
        m = Model()
        x = m.addVar(lb=-5.0)
        assert x.LB == pytest.approx(-5.0)

    def test_ub_custom(self):
        m = Model()
        x = m.addVar(ub=10.0)
        assert x.UB == pytest.approx(10.0)

    def test_vtype_continuous_default(self):
        m = Model()
        x = m.addVar()
        assert x.VType == GRB.CONTINUOUS

    def test_vtype_integer(self):
        m = Model()
        x = m.addVar(vtype=GRB.INTEGER)
        assert x.VType == GRB.INTEGER

    def test_vtype_binary(self):
        m = Model()
        x = m.addVar(vtype=GRB.BINARY)
        assert x.VType == GRB.BINARY

    def test_binary_bounds_are_0_1(self):
        m = Model()
        x = m.addVar(vtype=GRB.BINARY)
        assert x.LB == pytest.approx(0.0)
        assert x.UB == pytest.approx(1.0)

    def test_obj_coefficient(self):
        m = Model()
        x = m.addVar(obj=3.5)
        assert x.Obj == pytest.approx(3.5)

    def test_obj_default_zero(self):
        m = Model()
        x = m.addVar()
        assert x.Obj == pytest.approx(0.0)

    def test_repr(self):
        m = Model()
        x = m.addVar(name="foo")
        assert "foo" in repr(x)


# ------------------------------------------------------------------ #
# Attribute setters
# ------------------------------------------------------------------ #

class TestVarSetters:
    def test_lb_setter(self):
        m = Model()
        x = m.addVar(lb=0.0, ub=10.0)
        x.LB = 2.0
        assert x.LB == pytest.approx(2.0)

    def test_ub_setter(self):
        m = Model()
        x = m.addVar(lb=0.0, ub=10.0)
        x.UB = 7.0
        assert x.UB == pytest.approx(7.0)

    def test_obj_setter(self):
        m = Model()
        x = m.addVar(obj=1.0)
        x.Obj = 5.0
        assert x.Obj == pytest.approx(5.0)

    def test_vtype_setter_to_integer(self):
        m = Model()
        x = m.addVar(vtype=GRB.CONTINUOUS)
        x.VType = GRB.INTEGER
        assert x.VType == GRB.INTEGER

    def test_vtype_setter_to_binary(self):
        m = Model()
        x = m.addVar(vtype=GRB.CONTINUOUS, ub=GRB.INFINITY)
        x.VType = GRB.BINARY
        assert x.VType == GRB.BINARY
        assert x.LB == pytest.approx(0.0)
        assert x.UB == pytest.approx(1.0)

    def test_vtype_setter_back_to_continuous(self):
        m = Model()
        x = m.addVar(vtype=GRB.INTEGER)
        x.VType = GRB.CONTINUOUS
        assert x.VType == GRB.CONTINUOUS

    def test_lb_setter_takes_effect_in_optimize(self):
        m = Model()
        x = m.addVar(lb=0.0)
        x.LB = 3.0
        m.setObjective(x, GRB.MINIMIZE)
        m.optimize()
        assert m.Status == GRB.OPTIMAL
        assert x.X == pytest.approx(3.0)

    def test_ub_setter_takes_effect_in_optimize(self):
        m = Model()
        x = m.addVar(lb=0.0, ub=10.0)
        x.UB = 4.0
        m.setObjective(x, GRB.MAXIMIZE)
        m.optimize()
        assert x.X == pytest.approx(4.0)

    def test_obj_setter_takes_effect_in_optimize(self):
        m = Model()
        x = m.addVar(lb=0.0)
        y = m.addVar(lb=0.0)
        m.addConstr(x + y == 3)
        m.setObjective(x + y, GRB.MINIMIZE)  # same cost initially
        x.Obj = 10.0  # now x is expensive
        y.Obj = 1.0
        m.optimize()
        assert y.X == pytest.approx(3.0)
        assert x.X == pytest.approx(0.0)


# ------------------------------------------------------------------ #
# Solution attributes
# ------------------------------------------------------------------ #

class TestVarSolutionAttributes:
    def test_X_raises_before_optimize(self):
        m = Model()
        x = m.addVar()
        with pytest.raises(RuntimeError):
            _ = x.X

    def test_RC_raises_before_optimize(self):
        m = Model()
        x = m.addVar()
        with pytest.raises(RuntimeError):
            _ = x.RC

    def test_X_after_optimize(self):
        m, x, y = solved_model()
        assert x.X + y.X == pytest.approx(2.0)

    def test_X_non_negative(self):
        m, x, y = solved_model()
        assert x.X >= -1e-9
        assert y.X >= -1e-9

    def test_RC_after_optimize(self):
        m, x, y = solved_model()
        # reduced cost is defined for LP
        assert isinstance(x.RC, float)

    def test_X_clears_after_reset(self):
        m, x, _ = solved_model()
        m.reset()
        with pytest.raises(RuntimeError):
            _ = x.X


# ------------------------------------------------------------------ #
# Arithmetic operators
# ------------------------------------------------------------------ #

class TestVarArithmetic:
    def setup_method(self):
        self.m = Model()
        self.x = self.m.addVar(name="x")
        self.y = self.m.addVar(name="y")

    def test_add_var_var(self):
        e = self.x + self.y
        assert isinstance(e, LinExpr)
        assert e.size() == 2

    def test_add_var_number(self):
        e = self.x + 4
        assert e.size() == 1
        assert e.constant == pytest.approx(4.0)

    def test_radd_number_var(self):
        e = 4 + self.x
        assert e.size() == 1
        assert e.constant == pytest.approx(4.0)

    def test_radd_zero(self):
        e = 0 + self.x
        assert e.size() == 1

    def test_sub_var_var(self):
        e = self.x - self.y
        assert e.size() == 2
        terms = {e.getVar(i).VarName: e.getCoeff(i) for i in range(e.size())}
        assert terms["x"] == pytest.approx(1.0)
        assert terms["y"] == pytest.approx(-1.0)

    def test_sub_var_number(self):
        e = self.x - 3
        assert e.constant == pytest.approx(-3.0)

    def test_rsub_number_var(self):
        e = 5 - self.x
        assert e.getCoeff(0) == pytest.approx(-1.0)
        assert e.constant == pytest.approx(5.0)

    def test_mul_var_int(self):
        e = self.x * 3
        assert e.getCoeff(0) == pytest.approx(3.0)

    def test_rmul_int_var(self):
        e = 3 * self.x
        assert e.getCoeff(0) == pytest.approx(3.0)

    def test_mul_var_float(self):
        e = self.x * 2.5
        assert e.getCoeff(0) == pytest.approx(2.5)

    def test_div_var_scalar(self):
        e = self.x / 4
        assert e.getCoeff(0) == pytest.approx(0.25)

    def test_neg_var(self):
        e = -self.x
        assert e.getCoeff(0) == pytest.approx(-1.0)

    def test_mul_zero_gives_empty_expr(self):
        e = self.x * 0
        assert e.size() == 0

    def test_add_linexpr(self):
        base = 2 * self.y + 1
        result = self.x + base
        assert result.size() == 2
        assert result.constant == pytest.approx(1.0)

    def test_sub_linexpr(self):
        base = 2 * self.y + 1
        result = self.x - base
        assert result.size() == 2
        assert result.constant == pytest.approx(-1.0)

    def test_not_implemented_for_non_scalar(self):
        result = self.x.__mul__("string")
        assert result is NotImplemented


# ------------------------------------------------------------------ #
# Comparison operators → TempConstr
# ------------------------------------------------------------------ #

class TestVarComparisons:
    def setup_method(self):
        self.m = Model()
        self.x = self.m.addVar(name="x")
        self.y = self.m.addVar(name="y")

    def test_le_number(self):
        tc = self.x <= 5
        assert isinstance(tc, TempConstr)
        assert tc.sense == GRB.LESS_EQUAL
        assert tc.rhs == 5

    def test_ge_number(self):
        tc = self.x >= 3
        assert isinstance(tc, TempConstr)
        assert tc.sense == GRB.GREATER_EQUAL
        assert tc.rhs == 3

    def test_eq_number(self):
        tc = self.x == 7
        assert isinstance(tc, TempConstr)
        assert tc.sense == GRB.EQUAL
        assert tc.rhs == 7

    def test_le_var(self):
        tc = self.x <= self.y
        assert isinstance(tc, TempConstr)
        assert tc.sense == GRB.LESS_EQUAL
        assert tc.rhs is self.y

    def test_ge_var(self):
        tc = self.x >= self.y
        assert isinstance(tc, TempConstr)
        assert tc.sense == GRB.GREATER_EQUAL

    def test_eq_var(self):
        tc = self.x == self.y
        assert isinstance(tc, TempConstr)

    def test_le_linexpr(self):
        tc = self.x <= (2 * self.y + 1)
        assert isinstance(tc, TempConstr)

    def test_hash_identity_based(self):
        """Two distinct Var objects always have different hashes."""
        assert hash(self.x) != hash(self.y)
        assert hash(self.x) == id(self.x)

    def test_var_usable_as_dict_key(self):
        d = {self.x: 10, self.y: 20}
        assert d[self.x] == 10
        assert d[self.y] == 20

    def test_var_usable_in_set(self):
        s = {self.x, self.y}
        assert len(s) == 2
