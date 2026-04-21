"""Tests for Model construction and core model-building API."""

import os
import itertools
import tempfile

import pytest
from grbcompat import GRB, Model
from grbcompat._constr import Constr
from grbcompat._tupledict import tupledict
from grbcompat._var import Var


# ------------------------------------------------------------------ #
# Construction
# ------------------------------------------------------------------ #

class TestModelConstruction:
    def test_default_name(self):
        m = Model()
        assert m.ModelName == ""

    def test_custom_name(self):
        m = Model("my_model")
        assert m.ModelName == "my_model"

    def test_model_name_setter(self):
        m = Model()
        m.ModelName = "renamed"
        assert m.ModelName == "renamed"

    def test_initial_status_loaded(self):
        m = Model()
        assert m.Status == GRB.LOADED

    def test_initial_num_vars(self):
        m = Model()
        assert m.NumVars == 0

    def test_initial_num_constrs(self):
        m = Model()
        assert m.NumConstrs == 0

    def test_params_attribute_exists(self):
        from grbcompat._params import Params
        m = Model()
        assert isinstance(m.Params, Params)

    def test_env_parameter_accepted(self):
        """env= parameter must be silently accepted (gurobipy compat)."""
        m = Model("m", env=None)
        assert m.NumVars == 0

    def test_context_manager_enter_returns_model(self):
        with Model("ctx") as m:
            assert isinstance(m, Model)

    def test_context_manager_exit_no_error(self):
        with Model("ctx") as m:
            m.addVar()
        # No exception on __exit__

    def test_dispose_no_error(self):
        m = Model()
        m.dispose()  # should not raise

    def test_repr_contains_name(self):
        m = Model("test_repr")
        assert "test_repr" in repr(m)

    def test_repr_contains_var_count(self):
        m = Model()
        m.addVar()
        assert "vars=1" in repr(m)


# ------------------------------------------------------------------ #
# addVar
# ------------------------------------------------------------------ #

class TestAddVar:
    def test_returns_var(self):
        m = Model()
        v = m.addVar()
        assert isinstance(v, Var)

    def test_num_vars_increments(self):
        m = Model()
        m.addVar()
        assert m.NumVars == 1
        m.addVar()
        assert m.NumVars == 2

    def test_col_indices_sequential(self):
        m = Model()
        a = m.addVar()
        b = m.addVar()
        c = m.addVar()
        assert a._col_idx == 0
        assert b._col_idx == 1
        assert c._col_idx == 2

    def test_default_lb_zero(self):
        m = Model()
        x = m.addVar()
        assert x.LB == pytest.approx(0.0)

    def test_default_ub_infinity(self):
        m = Model()
        x = m.addVar()
        assert x.UB == float("inf")

    def test_custom_lb(self):
        m = Model()
        x = m.addVar(lb=-10.0)
        assert x.LB == pytest.approx(-10.0)

    def test_custom_ub(self):
        m = Model()
        x = m.addVar(ub=5.0)
        assert x.UB == pytest.approx(5.0)

    def test_negative_infinity_lb(self):
        m = Model()
        x = m.addVar(lb=-GRB.INFINITY)
        import math
        assert math.isinf(x.LB) and x.LB < 0

    def test_obj_coefficient(self):
        m = Model()
        x = m.addVar(obj=2.5)
        assert x.Obj == pytest.approx(2.5)

    def test_name(self):
        m = Model()
        x = m.addVar(name="alpha")
        assert x.VarName == "alpha"

    def test_continuous_type(self):
        m = Model()
        x = m.addVar(vtype=GRB.CONTINUOUS)
        assert x.VType == GRB.CONTINUOUS

    def test_integer_type(self):
        m = Model()
        x = m.addVar(vtype=GRB.INTEGER)
        assert x.VType == GRB.INTEGER

    def test_binary_type(self):
        m = Model()
        x = m.addVar(vtype=GRB.BINARY)
        assert x.VType == GRB.BINARY
        assert x.LB == pytest.approx(0.0)
        assert x.UB == pytest.approx(1.0)

    def test_getVars_returns_all(self):
        m = Model()
        vars_ = [m.addVar() for _ in range(4)]
        assert m.getVars() == vars_

    def test_getVarByName_found(self):
        m = Model()
        x = m.addVar(name="needle")
        m.addVar(name="other")
        assert m.getVarByName("needle") is x

    def test_getVarByName_not_found(self):
        m = Model()
        assert m.getVarByName("ghost") is None


# ------------------------------------------------------------------ #
# addVars
# ------------------------------------------------------------------ #

class TestAddVars:
    def test_single_int_returns_tupledict(self):
        m = Model()
        x = m.addVars(4)
        assert isinstance(x, tupledict)

    def test_single_int_count(self):
        m = Model()
        x = m.addVars(5)
        assert len(x) == 5

    def test_single_int_keys_are_integers(self):
        m = Model()
        x = m.addVars(3)
        assert set(x.keys()) == {0, 1, 2}

    def test_single_range_keys(self):
        m = Model()
        x = m.addVars(range(2, 5))
        assert set(x.keys()) == {2, 3, 4}

    def test_list_keys(self):
        m = Model()
        x = m.addVars(["a", "b", "c"])
        assert set(x.keys()) == {"a", "b", "c"}

    def test_two_ints_cartesian_product(self):
        m = Model()
        x = m.addVars(2, 3)
        expected = set(itertools.product(range(2), range(3)))
        assert set(x.keys()) == expected

    def test_two_lists_cartesian_product(self):
        m = Model()
        x = m.addVars(["i", "j"], [1, 2])
        expected = {("i", 1), ("i", 2), ("j", 1), ("j", 2)}
        assert set(x.keys()) == expected

    def test_names_follow_convention(self):
        m = Model()
        x = m.addVars(3, name="x")
        assert x[0].VarName == "x[0]"
        assert x[1].VarName == "x[1]"
        assert x[2].VarName == "x[2]"

    def test_2d_names(self):
        m = Model()
        x = m.addVars(2, 2, name="x")
        assert x[(0, 0)].VarName == "x[0,0]"
        assert x[(1, 1)].VarName == "x[1,1]"

    def test_lb_applies_to_all(self):
        m = Model()
        x = m.addVars(3, lb=-1.0)
        assert all(v.LB == pytest.approx(-1.0) for v in x.values())

    def test_ub_applies_to_all(self):
        m = Model()
        x = m.addVars(3, ub=7.0)
        assert all(v.UB == pytest.approx(7.0) for v in x.values())

    def test_vtype_binary_all(self):
        m = Model()
        x = m.addVars(3, vtype=GRB.BINARY)
        assert all(v.VType == GRB.BINARY for v in x.values())

    def test_lb_dict(self):
        m = Model()
        lb_map = {0: 0.0, 1: 1.0, 2: 2.0}
        x = m.addVars(3, lb=lb_map)
        assert x[0].LB == pytest.approx(0.0)
        assert x[1].LB == pytest.approx(1.0)
        assert x[2].LB == pytest.approx(2.0)

    def test_ub_dict(self):
        m = Model()
        ub_map = {0: 3.0, 1: 5.0, 2: 7.0}
        x = m.addVars(3, ub=ub_map)
        assert x[0].UB == pytest.approx(3.0)
        assert x[1].UB == pytest.approx(5.0)
        assert x[2].UB == pytest.approx(7.0)

    def test_num_vars_after_addvars(self):
        m = Model()
        m.addVars(3, 4)
        assert m.NumVars == 12

    def test_no_args_raises(self):
        m = Model()
        with pytest.raises((ValueError, TypeError)):
            m.addVars()


# ------------------------------------------------------------------ #
# addConstr
# ------------------------------------------------------------------ #

class TestAddConstr:
    def test_returns_constr(self):
        m = Model()
        x = m.addVar()
        c = m.addConstr(x <= 5)
        assert isinstance(c, Constr)

    def test_num_constrs_increments(self):
        m = Model()
        x = m.addVar()
        m.addConstr(x <= 5)
        assert m.NumConstrs == 1
        m.addConstr(x >= 0)
        assert m.NumConstrs == 2

    def test_row_indices_sequential(self):
        m = Model()
        x = m.addVar()
        c0 = m.addConstr(x <= 5)
        c1 = m.addConstr(x >= 1)
        assert c0._row_idx == 0
        assert c1._row_idx == 1

    def test_le_from_tempconstr(self):
        m = Model()
        x = m.addVar()
        c = m.addConstr(x <= 10, name="cap")
        assert c.Sense == GRB.LESS_EQUAL
        assert c.RHS == pytest.approx(10.0)

    def test_ge_from_tempconstr(self):
        m = Model()
        x = m.addVar()
        c = m.addConstr(x >= 2)
        assert c.Sense == GRB.GREATER_EQUAL
        assert c.RHS == pytest.approx(2.0)

    def test_eq_from_tempconstr(self):
        m = Model()
        x = m.addVar()
        c = m.addConstr(x == 3)
        assert c.Sense == GRB.EQUAL
        assert c.RHS == pytest.approx(3.0)

    def test_explicit_sense_and_rhs(self):
        m = Model()
        x = m.addVar()
        c = m.addConstr(x, GRB.LESS_EQUAL, 7)
        assert c.Sense == GRB.LESS_EQUAL
        assert c.RHS == pytest.approx(7.0)

    def test_name_stored(self):
        m = Model()
        x = m.addVar()
        c = m.addConstr(x <= 5, name="my_c")
        assert c.ConstrName == "my_c"

    def test_linexpr_on_lhs(self):
        m = Model()
        x = m.addVar()
        y = m.addVar()
        c = m.addConstr(x + y <= 8, name="sum")
        assert c.RHS == pytest.approx(8.0)

    def test_constant_on_lhs_absorbed_into_rhs(self):
        """x + 2 <= 8  →  x <= 6."""
        m = Model()
        x = m.addVar(lb=0.0)
        c = m.addConstr(x + 2 <= 8)
        m.setObjective(x, GRB.MAXIMIZE)
        m.optimize()
        assert x.X == pytest.approx(6.0, abs=1e-6)

    def test_var_on_rhs_moved_to_lhs(self):
        """x <= y  is equivalent to  x - y <= 0."""
        m = Model()
        x = m.addVar(lb=0.0, ub=3.0)
        y = m.addVar(lb=0.0, ub=5.0)
        m.addConstr(x <= y)
        m.setObjective(x - y, GRB.MAXIMIZE)
        m.optimize()
        assert m.Status == GRB.OPTIMAL
        assert x.X <= y.X + 1e-6

    def test_linexpr_on_rhs(self):
        m = Model()
        x = m.addVar(lb=0.0, ub=10.0)
        y = m.addVar(lb=0.0, ub=10.0)
        m.addConstr(x <= y + 3)
        m.setObjective(x - y, GRB.MAXIMIZE)
        m.optimize()
        assert m.Status == GRB.OPTIMAL
        assert x.X <= y.X + 3 + 1e-6

    def test_unknown_sense_raises(self):
        m = Model()
        x = m.addVar()
        with pytest.raises((ValueError, Exception)):
            m.addConstr(x, "?", 5)

    def test_getConstrs_returns_all(self):
        m = Model()
        x = m.addVar()
        c0 = m.addConstr(x <= 5)
        c1 = m.addConstr(x >= 1)
        assert m.getConstrs() == [c0, c1]

    def test_getConstrByName_found(self):
        m = Model()
        x = m.addVar()
        c = m.addConstr(x <= 5, name="target")
        assert m.getConstrByName("target") is c

    def test_getConstrByName_not_found(self):
        m = Model()
        assert m.getConstrByName("ghost") is None

    def test_addLConstr_alias(self):
        m = Model()
        x = m.addVar()
        c = m.addLConstr(x <= 5, name="lc")
        assert isinstance(c, Constr)


# ------------------------------------------------------------------ #
# addConstrs
# ------------------------------------------------------------------ #

class TestAddConstrs:
    def test_returns_tupledict(self):
        m = Model()
        x = m.addVars(3)
        result = m.addConstrs((x[i] <= i + 1 for i in range(3)))
        assert isinstance(result, tupledict)

    def test_count_matches_generator(self):
        m = Model()
        x = m.addVars(5)
        cs = m.addConstrs((x[i] <= i + 1 for i in range(5)))
        assert len(cs) == 5
        assert m.NumConstrs == 5

    def test_sequential_indices(self):
        m = Model()
        x = m.addVars(3)
        cs = m.addConstrs((x[i] <= 5 for i in range(3)))
        assert set(cs.keys()) == {0, 1, 2}

    def test_names_follow_convention(self):
        m = Model()
        x = m.addVars(3)
        cs = m.addConstrs((x[i] <= 5 for i in range(3)), name="cap")
        assert cs[0].ConstrName == "cap[0]"
        assert cs[2].ConstrName == "cap[2]"


# ------------------------------------------------------------------ #
# setObjective
# ------------------------------------------------------------------ #

class TestSetObjective:
    def test_sets_minimize_sense(self):
        m = Model()
        x = m.addVar()
        m.setObjective(x, GRB.MINIMIZE)
        assert m.ModelSense == GRB.MINIMIZE

    def test_sets_maximize_sense(self):
        m = Model()
        x = m.addVar()
        m.setObjective(x, GRB.MAXIMIZE)
        assert m.ModelSense == GRB.MAXIMIZE

    def test_no_sense_keeps_existing(self):
        m = Model()
        x = m.addVar()
        m.setObjective(x, GRB.MAXIMIZE)
        m.setObjective(x + 1)
        assert m.ModelSense == GRB.MAXIMIZE

    def test_objective_constant_offset(self):
        """Objective constant term should be added to ObjVal."""
        m = Model()
        x = m.addVar(lb=2.0, ub=2.0)
        m.setObjective(x + 10, GRB.MINIMIZE)
        m.optimize()
        assert m.ObjVal == pytest.approx(12.0, abs=1e-6)

    def test_setAttr_model_sense_minimize(self):
        m = Model()
        x = m.addVar()
        m.setObjective(x)
        m.setAttr("ModelSense", GRB.MINIMIZE)
        assert m.ModelSense == GRB.MINIMIZE

    def test_setAttr_model_sense_maximize(self):
        m = Model()
        x = m.addVar()
        m.setObjective(x)
        m.setAttr("ModelSense", GRB.MAXIMIZE)
        assert m.ModelSense == GRB.MAXIMIZE

    def test_modelsense_property_setter(self):
        m = Model()
        x = m.addVar()
        m.setObjective(x)
        m.ModelSense = GRB.MAXIMIZE
        assert m.ModelSense == GRB.MAXIMIZE


# ------------------------------------------------------------------ #
# update / reset
# ------------------------------------------------------------------ #

class TestUpdateAndReset:
    def test_update_is_noop(self):
        m = Model()
        x = m.addVar()
        m.update()  # should not raise
        assert m.NumVars == 1

    def test_reset_clears_solution(self):
        m = Model()
        x = m.addVar(lb=0.0)
        m.setObjective(x, GRB.MINIMIZE)
        m.optimize()
        assert m.Status == GRB.OPTIMAL
        m.reset()
        assert m.Status == GRB.LOADED
        with pytest.raises(RuntimeError):
            _ = x.X


# ------------------------------------------------------------------ #
# getAttr
# ------------------------------------------------------------------ #

class TestGetAttr:
    def test_getAttr_obj_val(self):
        m = Model()
        x = m.addVar(lb=1.0, ub=1.0)
        m.setObjective(x, GRB.MINIMIZE)
        m.optimize()
        assert m.getAttr("ObjVal") == pytest.approx(1.0)

    def test_getAttr_var_X(self):
        m = Model()
        x = m.addVar(lb=2.0, ub=2.0)
        y = m.addVar(lb=3.0, ub=3.0)
        m.setObjective(x + y, GRB.MINIMIZE)
        m.optimize()
        xs = m.getAttr("X", [x, y])
        assert xs == pytest.approx([2.0, 3.0])

    def test_getAttr_constr_Pi(self):
        m = Model()
        x = m.addVar(lb=0.0)
        c = m.addConstr(x >= 3)
        m.setObjective(x, GRB.MINIMIZE)
        m.optimize()
        pis = m.getAttr("Pi", [c])
        assert len(pis) == 1
        assert isinstance(pis[0], float)

    def test_getAttr_unknown_raises(self):
        m = Model()
        with pytest.raises(AttributeError):
            m.getAttr("NoSuchAttr")


# ------------------------------------------------------------------ #
# I/O
# ------------------------------------------------------------------ #

class TestModelIO:
    def test_write_lp_creates_file(self):
        m = Model()
        x = m.addVar(lb=0.0, name="x")
        m.setObjective(x, GRB.MINIMIZE)
        m.addConstr(x >= 1)
        with tempfile.NamedTemporaryFile(suffix=".lp", delete=False) as f:
            path = f.name
        try:
            m.write(path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)

    def test_write_mps_creates_file(self):
        m = Model()
        x = m.addVar(lb=0.0)
        m.setObjective(x, GRB.MINIMIZE)
        m.addConstr(x >= 1)
        with tempfile.NamedTemporaryFile(suffix=".mps", delete=False) as f:
            path = f.name
        try:
            m.write(path)
            assert os.path.exists(path)
        finally:
            os.unlink(path)
