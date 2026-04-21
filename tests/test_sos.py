"""Tests for SOS1 constraints via big-M linearization."""

import pytest
from grbcompat import GRB, Model
from grbcompat._sos import SOS


TOL = 1e-6


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _sos1_model():
    """
    max  3*x0 + 5*x1 + 2*x2
    s.t. SOS1({x0, x1, x2})
         0 <= x_i <= 10 for all i
    Optimal: x1 = 10, obj = 50.
    """
    m = Model()
    x0 = m.addVar(lb=0.0, ub=10.0, name="x0")
    x1 = m.addVar(lb=0.0, ub=10.0, name="x1")
    x2 = m.addVar(lb=0.0, ub=10.0, name="x2")
    m.setObjective(3 * x0 + 5 * x1 + 2 * x2, GRB.MAXIMIZE)
    sos = m.addSOS(GRB.SOS_TYPE1, [x0, x1, x2])
    return m, x0, x1, x2, sos


def _sos1_min_model():
    """
    min  3*x0 + 1*x1 + 4*x2
    s.t. x0 + x1 + x2 >= 5
         SOS1({x0, x1, x2})
         0 <= x_i <= 10 for all i
    Optimal: x1 = 5, obj = 5.
    """
    m = Model()
    x0 = m.addVar(lb=0.0, ub=10.0, name="x0")
    x1 = m.addVar(lb=0.0, ub=10.0, name="x1")
    x2 = m.addVar(lb=0.0, ub=10.0, name="x2")
    m.addConstr(x0 + x1 + x2 >= 5)
    m.setObjective(3 * x0 + 1 * x1 + 4 * x2, GRB.MINIMIZE)
    m.addSOS(GRB.SOS_TYPE1, [x0, x1, x2])
    return m, x0, x1, x2


# ------------------------------------------------------------------ #
# GRB constants
# ------------------------------------------------------------------ #

class TestSOSConstants:
    def test_sos_type1_value(self):
        assert GRB.SOS_TYPE1 == 1

    def test_sos_type2_value(self):
        assert GRB.SOS_TYPE2 == 2


# ------------------------------------------------------------------ #
# addSOS validation
# ------------------------------------------------------------------ #

class TestAddSOS:
    def test_returns_sos_object(self):
        m, *_, sos = _sos1_model()
        assert isinstance(sos, SOS)

    def test_sos_type2_raises(self):
        m = Model()
        x = m.addVar(lb=0.0, ub=5.0)
        with pytest.raises(NotImplementedError):
            m.addSOS(GRB.SOS_TYPE2, [x])

    def test_unknown_type_raises(self):
        m = Model()
        x = m.addVar(lb=0.0, ub=5.0)
        with pytest.raises(ValueError):
            m.addSOS(99, [x])

    def test_unbounded_var_raises(self):
        m = Model()
        x = m.addVar(lb=0.0)  # UB = GRB.INFINITY
        with pytest.raises(ValueError, match="finite upper bound"):
            m.addSOS(GRB.SOS_TYPE1, [x])

    def test_default_weights(self):
        m = Model()
        x0 = m.addVar(lb=0.0, ub=1.0)
        x1 = m.addVar(lb=0.0, ub=1.0)
        x2 = m.addVar(lb=0.0, ub=1.0)
        sos = m.addSOS(GRB.SOS_TYPE1, [x0, x1, x2])
        assert sos._weights == [1, 2, 3]

    def test_explicit_weights_stored(self):
        m = Model()
        x0 = m.addVar(lb=0.0, ub=1.0)
        x1 = m.addVar(lb=0.0, ub=1.0)
        sos = m.addSOS(GRB.SOS_TYPE1, [x0, x1], weights=[10.0, 20.0])
        assert sos._weights == [10.0, 20.0]

    def test_wrong_weights_length_raises(self):
        m = Model()
        x0 = m.addVar(lb=0.0, ub=1.0)
        x1 = m.addVar(lb=0.0, ub=1.0)
        with pytest.raises(ValueError):
            m.addSOS(GRB.SOS_TYPE1, [x0, x1], weights=[1.0])

    def test_numsos_increments(self):
        m = Model()
        x0 = m.addVar(lb=0.0, ub=5.0)
        x1 = m.addVar(lb=0.0, ub=5.0)
        assert m.NumSOS == 0
        m.addSOS(GRB.SOS_TYPE1, [x0, x1])
        assert m.NumSOS == 1
        x2 = m.addVar(lb=0.0, ub=5.0)
        m.addSOS(GRB.SOS_TYPE1, [x0, x2])
        assert m.NumSOS == 2

    def test_getSOSs_returns_list(self):
        m, *_, sos = _sos1_model()
        result = m.getSOSs()
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] is sos

    def test_helper_vars_count(self):
        """n member vars → n binary helper vars."""
        m, *_, sos = _sos1_model()
        assert len(sos._helper_vars) == 3

    def test_helper_constrs_count(self):
        """n member vars → n big-M constrs + 1 sum constr."""
        m, *_, sos = _sos1_model()
        assert len(sos._helper_constrs) == 4  # 3 big-M + 1 sum

    def test_numvars_includes_helpers(self):
        """3 originals + 3 binaries = 6 total vars."""
        m, *_, sos = _sos1_model()
        assert m.NumVars == 6

    def test_numconstrs_includes_helpers(self):
        """3 big-M + 1 sum = 4 total constraints."""
        m, *_, sos = _sos1_model()
        assert m.NumConstrs == 4


# ------------------------------------------------------------------ #
# SOS1 correctness
# ------------------------------------------------------------------ #

class TestSOS1Correctness:
    def test_maximize_selects_highest_coeff_var(self):
        """x1 has highest obj coeff (5); SOS1 → only x1 nonzero."""
        m, x0, x1, x2, sos = _sos1_model()
        m.optimize()
        assert m.Status == GRB.OPTIMAL
        assert x1.X == pytest.approx(10.0, abs=TOL)
        assert x0.X == pytest.approx(0.0, abs=TOL)
        assert x2.X == pytest.approx(0.0, abs=TOL)

    def test_maximize_obj_value(self):
        m, *_ = _sos1_model()
        m.optimize()
        assert m.ObjVal == pytest.approx(50.0, abs=TOL)

    def test_minimize_selects_lowest_cost_var(self):
        """x1 has lowest cost (1); SOS1 → only x1 nonzero."""
        m, x0, x1, x2 = _sos1_min_model()
        m.optimize()
        assert m.Status == GRB.OPTIMAL
        assert x1.X == pytest.approx(5.0, abs=TOL)
        assert x0.X == pytest.approx(0.0, abs=TOL)
        assert x2.X == pytest.approx(0.0, abs=TOL)

    def test_minimize_obj_value(self):
        m, *_ = _sos1_min_model()
        m.optimize()
        assert m.ObjVal == pytest.approx(5.0, abs=TOL)

    def test_all_zero_is_feasible(self):
        """SOS1 allows all member variables to be zero."""
        m = Model()
        x0 = m.addVar(lb=0.0, ub=10.0)
        x1 = m.addVar(lb=0.0, ub=10.0)
        m.setObjective(x0 + x1, GRB.MINIMIZE)
        m.addSOS(GRB.SOS_TYPE1, [x0, x1])
        m.optimize()
        assert m.Status == GRB.OPTIMAL
        assert m.ObjVal == pytest.approx(0.0, abs=TOL)

    def test_two_var_sos1(self):
        """With two vars, only one can be nonzero."""
        m = Model()
        x = m.addVar(lb=0.0, ub=5.0, name="x")
        y = m.addVar(lb=0.0, ub=5.0, name="y")
        m.setObjective(x + y, GRB.MAXIMIZE)
        m.addSOS(GRB.SOS_TYPE1, [x, y])
        m.optimize()
        assert m.Status == GRB.OPTIMAL
        assert m.ObjVal == pytest.approx(5.0, abs=TOL)
        # Exactly one is 5, the other is 0
        vals = sorted([x.X, y.X])
        assert vals[0] == pytest.approx(0.0, abs=TOL)
        assert vals[1] == pytest.approx(5.0, abs=TOL)

    def test_named_sos(self):
        m = Model()
        x = m.addVar(lb=0.0, ub=3.0, name="x")
        y = m.addVar(lb=0.0, ub=3.0, name="y")
        sos = m.addSOS(GRB.SOS_TYPE1, [x, y], name="my_sos")
        assert sos._name == "my_sos"

    def test_sos_repr(self):
        m, *_, sos = _sos1_model()
        assert "SOS1" in repr(sos)


# ------------------------------------------------------------------ #
# SOS.index property
# ------------------------------------------------------------------ #

class TestSOSIndex:
    def test_first_sos_index_is_zero(self):
        m, *_, sos = _sos1_model()
        assert sos.index == 0

    def test_second_sos_index_is_one(self):
        m = Model()
        x0 = m.addVar(lb=0.0, ub=5.0)
        x1 = m.addVar(lb=0.0, ub=5.0)
        x2 = m.addVar(lb=0.0, ub=5.0)
        s0 = m.addSOS(GRB.SOS_TYPE1, [x0, x1])
        s1 = m.addSOS(GRB.SOS_TYPE1, [x1, x2])
        assert s0.index == 0
        assert s1.index == 1

    def test_removed_sos_index_raises(self):
        m, *_, sos = _sos1_model()
        m.remove(sos)
        with pytest.raises(RuntimeError):
            _ = sos.index


# ------------------------------------------------------------------ #
# remove(SOS)
# ------------------------------------------------------------------ #

class TestRemoveSOS:
    def test_numsos_decrements(self):
        m, *_, sos = _sos1_model()
        m.remove(sos)
        assert m.NumSOS == 0

    def test_numvars_decrements_by_helper_count(self):
        """Removing SOS deletes 3 binary helpers → 3 original vars remain."""
        m, x0, x1, x2, sos = _sos1_model()
        m.remove(sos)
        assert m.NumVars == 3

    def test_numconstrs_decrements_by_helper_count(self):
        """Removing SOS deletes 4 helper constraints."""
        m, *_, sos = _sos1_model()
        m.remove(sos)
        assert m.NumConstrs == 0

    def test_removed_sos_raises_on_index(self):
        m, *_, sos = _sos1_model()
        m.remove(sos)
        with pytest.raises(RuntimeError):
            _ = sos.index

    def test_getSOSs_empty_after_remove(self):
        m, *_, sos = _sos1_model()
        m.remove(sos)
        assert m.getSOSs() == []

    def test_remove_sos_then_solve_unrestricted(self):
        """After removing SOS, all vars can be nonzero simultaneously."""
        m, x0, x1, x2, sos = _sos1_model()
        m.remove(sos)
        m.optimize()
        assert m.Status == GRB.OPTIMAL
        # Without SOS1 all three can be at their UB of 10
        assert m.ObjVal == pytest.approx(100.0, abs=TOL)  # 3*10 + 5*10 + 2*10

    def test_remove_sos_from_mixed_list(self):
        m, x0, x1, x2, sos = _sos1_model()
        # Remove x2 (a member var) and the SOS in one call
        m.remove([x2, sos])
        assert m.NumSOS == 0
        assert x2._col_idx < 0

    def test_original_vars_survive_after_sos_removal(self):
        m, x0, x1, x2, sos = _sos1_model()
        m.remove(sos)
        survivors = m.getVars()
        assert any(v is x0 for v in survivors)
        assert any(v is x1 for v in survivors)
        assert any(v is x2 for v in survivors)


# ------------------------------------------------------------------ #
# copy() with SOS
# ------------------------------------------------------------------ #

class TestCopySOS:
    def test_copy_preserves_numsos(self):
        m, *_, sos = _sos1_model()
        cp = m.copy()
        assert cp.NumSOS == 1

    def test_copy_sos_is_independent_object(self):
        m, *_, sos = _sos1_model()
        cp = m.copy()
        assert cp.getSOSs()[0] is not sos

    def test_copy_sos_solves_correctly(self):
        m, *_ = _sos1_model()
        cp = m.copy()
        cp.optimize()
        assert cp.Status == GRB.OPTIMAL
        assert cp.ObjVal == pytest.approx(50.0, abs=TOL)

    def test_copy_sos_independent_remove(self):
        """Removing SOS from copy does not affect original."""
        m, *_, sos = _sos1_model()
        cp = m.copy()
        cp.remove(cp.getSOSs()[0])
        assert m.NumSOS == 1
        assert cp.NumSOS == 0

    def test_copy_sos_index_valid(self):
        m, *_, sos = _sos1_model()
        cp = m.copy()
        assert cp.getSOSs()[0].index == 0

    def test_copy_sos_preserves_name(self):
        m = Model()
        x = m.addVar(lb=0.0, ub=5.0)
        y = m.addVar(lb=0.0, ub=5.0)
        m.addSOS(GRB.SOS_TYPE1, [x, y], name="named_sos")
        cp = m.copy()
        assert cp.getSOSs()[0]._name == "named_sos"

    def test_copy_numvars_matches(self):
        m, *_ = _sos1_model()
        cp = m.copy()
        assert cp.NumVars == m.NumVars

    def test_copy_numconstrs_matches(self):
        m, *_ = _sos1_model()
        cp = m.copy()
        assert cp.NumConstrs == m.NumConstrs
