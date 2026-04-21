"""Tests for LinExpr and TempConstr."""

import pytest
from grbcompat import GRB, Model
from grbcompat._expr import LinExpr, TempConstr


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def make_vars(n=3):
    m = Model()
    return m, [m.addVar(name=f"x{i}") for i in range(n)]


# ------------------------------------------------------------------ #
# LinExpr construction
# ------------------------------------------------------------------ #

class TestLinExprConstruction:
    def test_empty_expr_has_zero_constant(self):
        e = LinExpr()
        assert e.constant == 0.0
        assert e.size() == 0

    def test_empty_expr_with_explicit_constant(self):
        e = LinExpr(constant=5.0)
        assert e.constant == 5.0
        assert e.size() == 0

    def test_construction_with_vars_and_coeffs(self):
        _, vars_ = make_vars(2)
        x, y = vars_
        e = LinExpr([2.0, 3.0], [x, y])
        assert e.size() == 2
        assert e.constant == 0.0

    def test_construction_scalar_only(self):
        # LinExpr(scalar) treats first arg as constant when no vars given
        e = LinExpr(7.0)
        assert e.constant == 7.0
        assert e.size() == 0

    def test_getCoeff_and_getVar(self):
        _, vars_ = make_vars(2)
        x, y = vars_
        e = LinExpr([4.0, -1.0], [x, y])
        # Order may vary; collect as a mapping
        terms = {e.getVar(i).VarName: e.getCoeff(i) for i in range(e.size())}
        assert terms["x0"] == pytest.approx(4.0)
        assert terms["x1"] == pytest.approx(-1.0)

    def test_getConstant(self):
        e = LinExpr(constant=3.14)
        assert e.getConstant() == pytest.approx(3.14)


# ------------------------------------------------------------------ #
# Arithmetic with Var objects
# ------------------------------------------------------------------ #

class TestLinExprArithmeticWithVar:
    def setup_method(self):
        self.m, vars_ = make_vars(3)
        self.x, self.y, self.z = vars_

    def test_var_plus_var(self):
        e = self.x + self.y
        assert isinstance(e, LinExpr)
        assert e.size() == 2
        assert e.constant == 0.0

    def test_var_plus_number(self):
        e = self.x + 5
        assert e.size() == 1
        assert e.constant == pytest.approx(5.0)

    def test_number_plus_var(self):
        e = 5 + self.x
        assert e.size() == 1
        assert e.constant == pytest.approx(5.0)

    def test_var_minus_var(self):
        e = self.x - self.y
        assert e.size() == 2

    def test_var_minus_number(self):
        e = self.x - 3
        assert e.size() == 1
        assert e.constant == pytest.approx(-3.0)

    def test_number_minus_var(self):
        e = 10 - self.x
        assert e.size() == 1
        assert e.constant == pytest.approx(10.0)
        # coefficient of x should be -1
        assert e.getCoeff(0) == pytest.approx(-1.0)

    def test_var_times_scalar(self):
        e = self.x * 3
        assert e.size() == 1
        assert e.getCoeff(0) == pytest.approx(3.0)

    def test_scalar_times_var(self):
        e = 3 * self.x
        assert e.getCoeff(0) == pytest.approx(3.0)

    def test_var_div_scalar(self):
        e = self.x / 4
        assert e.getCoeff(0) == pytest.approx(0.25)

    def test_neg_var(self):
        e = -self.x
        assert e.getCoeff(0) == pytest.approx(-1.0)

    def test_float_coefficient(self):
        e = 2.5 * self.x
        assert e.getCoeff(0) == pytest.approx(2.5)


# ------------------------------------------------------------------ #
# Arithmetic between LinExpr objects
# ------------------------------------------------------------------ #

class TestLinExprArithmetic:
    def setup_method(self):
        self.m, vars_ = make_vars(3)
        self.x, self.y, self.z = vars_

    def test_add_two_exprs(self):
        e1 = self.x + self.y
        e2 = self.y + self.z
        result = e1 + e2
        assert result.size() == 3

    def test_add_expr_and_number(self):
        e = self.x + self.y
        result = e + 7.0
        assert result.constant == pytest.approx(7.0)
        assert result.size() == 2

    def test_radd_zero_returns_copy(self):
        e = 2 * self.x + 3
        result = 0 + e
        assert result.size() == e.size()
        assert result.constant == e.constant

    def test_sum_builtin_on_vars(self):
        # sum([x, y, z]) uses __radd__ with 0 as start
        result = sum([self.x, self.y, self.z])
        assert isinstance(result, LinExpr)
        assert result.size() == 3

    def test_sub_two_exprs(self):
        e1 = 2 * self.x + 3 * self.y
        e2 = self.x + self.y
        result = e1 - e2
        assert result.size() == 2
        terms = {result.getVar(i).VarName: result.getCoeff(i) for i in range(result.size())}
        assert terms["x0"] == pytest.approx(1.0)
        assert terms["x1"] == pytest.approx(2.0)

    def test_sub_number_from_expr(self):
        e = self.x + 10
        result = e - 4
        assert result.constant == pytest.approx(6.0)

    def test_multiply_expr_by_scalar(self):
        e = self.x + 2 * self.y + 3
        result = e * 2
        assert result.constant == pytest.approx(6.0)
        terms = {result.getVar(i).VarName: result.getCoeff(i) for i in range(result.size())}
        assert terms["x0"] == pytest.approx(2.0)
        assert terms["x1"] == pytest.approx(4.0)

    def test_rmul_expr(self):
        e = self.x + self.y
        result = 3 * e
        terms = {result.getVar(i).VarName: result.getCoeff(i) for i in range(result.size())}
        assert all(abs(c - 3.0) < 1e-9 for c in terms.values())

    def test_div_expr_by_scalar(self):
        e = 4 * self.x
        result = e / 2
        assert result.getCoeff(0) == pytest.approx(2.0)

    def test_neg_expr(self):
        e = self.x + 2 * self.y + 1
        neg = -e
        assert neg.constant == pytest.approx(-1.0)
        terms = {neg.getVar(i).VarName: neg.getCoeff(i) for i in range(neg.size())}
        assert terms["x0"] == pytest.approx(-1.0)
        assert terms["x1"] == pytest.approx(-2.0)

    def test_same_var_coefficients_merge(self):
        """x + x should give 2x, not two separate terms."""
        e = self.x + self.x
        assert e.size() == 1
        assert e.getCoeff(0) == pytest.approx(2.0)

    def test_opposite_coefficients_cancel(self):
        """x - x should give empty expression (size 0)."""
        e = self.x - self.x
        assert e.size() == 0

    def test_add_linexpr_to_var(self):
        base = 2 * self.y + 5
        result = self.x + base
        assert result.size() == 2
        assert result.constant == pytest.approx(5.0)

    def test_constant_propagation(self):
        e1 = LinExpr(constant=3.0)
        e2 = LinExpr(constant=4.0)
        result = e1 + e2
        assert result.constant == pytest.approx(7.0)
        assert result.size() == 0

    def test_add_method_in_place(self):
        e = LinExpr()
        e.add(self.x, mult=2.0)
        assert e.size() == 1
        assert e.getCoeff(0) == pytest.approx(2.0)

    def test_add_method_with_linexpr(self):
        e = LinExpr()
        other = self.x + self.y
        e.add(other, mult=3.0)
        assert e.size() == 2
        terms = {e.getVar(i).VarName: e.getCoeff(i) for i in range(e.size())}
        assert terms["x0"] == pytest.approx(3.0)
        assert terms["x1"] == pytest.approx(3.0)

    def test_add_method_with_scalar(self):
        e = LinExpr(constant=1.0)
        e.add(5.0)
        assert e.constant == pytest.approx(6.0)

    def test_addTerms(self):
        e = LinExpr()
        e.addTerms([1.0, 2.0], [self.x, self.y])
        assert e.size() == 2

    def test_addTerms_single(self):
        e = LinExpr()
        e.addTerms(3.0, self.x)
        assert e.size() == 1
        assert e.getCoeff(0) == pytest.approx(3.0)

    def test_copy_is_independent(self):
        e1 = self.x + self.y
        e2 = e1._copy()
        e1.add(self.z)
        assert e2.size() == 2  # unaffected


# ------------------------------------------------------------------ #
# Comparison operators → TempConstr
# ------------------------------------------------------------------ #

class TestLinExprComparisons:
    def setup_method(self):
        self.m, vars_ = make_vars(2)
        self.x, self.y = vars_

    def test_le_returns_tempconstr(self):
        tc = (self.x + self.y) <= 5
        assert isinstance(tc, TempConstr)
        assert tc.sense == GRB.LESS_EQUAL
        assert tc.rhs == 5

    def test_ge_returns_tempconstr(self):
        tc = (self.x + self.y) >= 3
        assert isinstance(tc, TempConstr)
        assert tc.sense == GRB.GREATER_EQUAL

    def test_eq_returns_tempconstr(self):
        tc = (self.x + self.y) == 4
        assert isinstance(tc, TempConstr)
        assert tc.sense == GRB.EQUAL
        assert tc.rhs == 4

    def test_le_with_var_rhs(self):
        tc = self.x <= self.y
        assert isinstance(tc, TempConstr)
        assert tc.sense == GRB.LESS_EQUAL

    def test_le_with_linexpr_rhs(self):
        expr_rhs = 2 * self.y + 1
        tc = self.x <= expr_rhs
        assert isinstance(tc, TempConstr)

    def test_hash_is_identity_based(self):
        e1 = self.x + self.y
        e2 = self.x + self.y
        # Two distinct objects even if logically equal
        assert hash(e1) == id(e1)
        assert hash(e2) == id(e2)
        assert hash(e1) != hash(e2)

    def test_expr_usable_as_dict_key(self):
        e = self.x + self.y
        d = {e: "value"}
        assert d[e] == "value"


# ------------------------------------------------------------------ #
# TempConstr
# ------------------------------------------------------------------ #

class TestTempConstr:
    def setup_method(self):
        self.m, vars_ = make_vars(2)
        self.x, self.y = vars_

    def test_attributes_stored_correctly(self):
        lhs = 2 * self.x
        tc = TempConstr(lhs, GRB.LESS_EQUAL, 10)
        assert tc.lhs is lhs
        assert tc.sense == GRB.LESS_EQUAL
        assert tc.rhs == 10

    def test_var_comparison_produces_tempconstr(self):
        tc = self.x <= 5
        assert isinstance(tc, TempConstr)
        assert tc.sense == GRB.LESS_EQUAL
        assert tc.rhs == 5

    def test_var_eq_var(self):
        tc = self.x == self.y
        assert isinstance(tc, TempConstr)
        assert tc.sense == GRB.EQUAL
        assert tc.rhs is self.y
