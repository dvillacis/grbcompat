"""Tests for tupledict."""

import pytest
from grbcompat import Model
from grbcompat._expr import LinExpr
from grbcompat._tupledict import tupledict


# ------------------------------------------------------------------ #
# Basic dict behaviour
# ------------------------------------------------------------------ #

class TestTupledictBasicDict:
    def test_is_dict_subclass(self):
        td = tupledict()
        assert isinstance(td, dict)

    def test_set_and_get_scalar_key(self):
        td = tupledict()
        td[0] = "a"
        assert td[0] == "a"

    def test_set_and_get_tuple_key(self):
        td = tupledict()
        td[(0, 1)] = "b"
        assert td[(0, 1)] == "b"

    def test_len(self):
        td = tupledict({0: "a", 1: "b"})
        assert len(td) == 2

    def test_keys(self):
        td = tupledict({0: "a", 1: "b"})
        assert set(td.keys()) == {0, 1}

    def test_values(self):
        td = tupledict({0: 10, 1: 20})
        assert set(td.values()) == {10, 20}

    def test_items(self):
        td = tupledict({0: 10})
        assert list(td.items()) == [(0, 10)]

    def test_repr(self):
        td = tupledict({0: "x"})
        assert "tupledict" in repr(td)


# ------------------------------------------------------------------ #
# select()
# ------------------------------------------------------------------ #

class TestTupledictSelect:
    def setup_method(self):
        self.td = tupledict({
            (0, 0): "a", (0, 1): "b",
            (1, 0): "c", (1, 1): "d",
            (2, 0): "e",
        })

    def test_select_exact_match(self):
        result = self.td.select(0, 0)
        assert set(result.keys()) == {(0, 0)}

    def test_select_wildcard_first(self):
        result = self.td.select("*", 0)
        assert set(result.keys()) == {(0, 0), (1, 0), (2, 0)}

    def test_select_wildcard_second(self):
        result = self.td.select(0, "*")
        assert set(result.keys()) == {(0, 0), (0, 1)}

    def test_select_all_wildcard(self):
        result = self.td.select("*", "*")
        assert len(result) == len(self.td)

    def test_select_no_match(self):
        result = self.td.select(5, 5)
        assert len(result) == 0

    def test_select_wrong_dimension_excluded(self):
        # Entries with scalar keys don't match 2-element patterns
        td2 = tupledict({0: "x", (0, 1): "y"})
        result = td2.select(0, "*")
        assert set(result.keys()) == {(0, 1)}

    def test_select_returns_tupledict(self):
        result = self.td.select("*", 0)
        assert isinstance(result, tupledict)

    def test_select_with_scalar_keys(self):
        td = tupledict({0: "a", 1: "b", 2: "c"})
        result = td.select(1)
        assert set(result.keys()) == {1}

    def test_select_preserves_values(self):
        result = self.td.select(1, "*")
        assert result[(1, 0)] == "c"
        assert result[(1, 1)] == "d"


# ------------------------------------------------------------------ #
# sum()
# ------------------------------------------------------------------ #

class TestTupledictSum:
    def setup_method(self):
        self.m = Model()
        self.vars = self.m.addVars(3, 3, name="x")

    def test_sum_all(self):
        result = self.vars.sum()
        assert isinstance(result, LinExpr)
        assert result.size() == 9

    def test_sum_with_pattern_first_dim(self):
        result = self.vars.sum(0, "*")
        assert result.size() == 3  # (0,0), (0,1), (0,2)

    def test_sum_with_pattern_second_dim(self):
        result = self.vars.sum("*", 1)
        assert result.size() == 3  # (0,1), (1,1), (2,1)

    def test_sum_no_match(self):
        result = self.vars.sum(5, "*")
        assert result.size() == 0

    def test_sum_scalar_keys(self):
        m = Model()
        vars1d = m.addVars(4, name="v")
        result = vars1d.sum()
        assert result.size() == 4


# ------------------------------------------------------------------ #
# prod()
# ------------------------------------------------------------------ #

class TestTupledictProd:
    def test_prod_all(self):
        m = Model()
        x = m.addVars(3, name="x")
        coeffs = {0: 1.0, 1: 2.0, 2: 3.0}
        result = x.prod(coeffs)
        assert isinstance(result, LinExpr)
        assert result.size() == 3
        terms = {result.getVar(i).VarName: result.getCoeff(i) for i in range(result.size())}
        assert terms["x[0]"] == pytest.approx(1.0)
        assert terms["x[1]"] == pytest.approx(2.0)
        assert terms["x[2]"] == pytest.approx(3.0)

    def test_prod_with_pattern(self):
        m = Model()
        x = m.addVars(2, 2, name="x")
        coeffs = {(0, 0): 1.0, (0, 1): 2.0, (1, 0): 3.0, (1, 1): 4.0}
        result = x.prod(coeffs, 0, "*")
        assert result.size() == 2

    def test_prod_missing_key_skipped(self):
        m = Model()
        x = m.addVars(3, name="x")
        coeffs = {0: 5.0}  # only key 0
        result = x.prod(coeffs)
        assert result.size() == 1

    def test_prod_empty_coeffs(self):
        m = Model()
        x = m.addVars(3, name="x")
        result = x.prod({})
        assert result.size() == 0


# ------------------------------------------------------------------ #
# Integration with addVars
# ------------------------------------------------------------------ #

class TestTupledictFromAddVars:
    def test_scalar_keys_for_1d(self):
        m = Model()
        x = m.addVars(3, name="x")
        assert set(x.keys()) == {0, 1, 2}

    def test_tuple_keys_for_2d(self):
        m = Model()
        x = m.addVars(2, 3, name="x")
        expected = {(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)}
        assert set(x.keys()) == expected

    def test_list_index_keys(self):
        m = Model()
        x = m.addVars(["a", "b", "c"], name="x")
        assert set(x.keys()) == {"a", "b", "c"}

    def test_access_by_key(self):
        m = Model()
        x = m.addVars(3, name="x")
        from grbcompat._var import Var
        assert isinstance(x[0], Var)
        assert isinstance(x[2], Var)

    def test_2d_access_by_tuple(self):
        m = Model()
        x = m.addVars(2, 2, name="x")
        from grbcompat._var import Var
        assert isinstance(x[(0, 0)], Var)
        assert isinstance(x[(1, 1)], Var)
