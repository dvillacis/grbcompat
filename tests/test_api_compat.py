"""
gurobipy API compatibility tests.

These tests exercise the wrapper using ``grbcompat`` directly (no
sys.modules patching required). The ``TestInstallFunction`` class separately
verifies that ``install()`` correctly patches sys.modules without side-effects
on other tests.
"""

import importlib
import sys

import pytest

import grbcompat as gp
from grbcompat import GRB


TOL = 1e-6


# ------------------------------------------------------------------ #
# install() – sys.modules patching function
# ------------------------------------------------------------------ #

class TestInstallFunction:
    """
    Verify that ``grbcompat.install()`` correctly patches sys.modules
    so that ``import gurobipy`` resolves to the wrapper.

    Each test saves and fully restores the sys.modules state so that the
    comparison tests (which need real gurobipy) are not affected.
    """

    def setup_method(self):
        """Save current sys.modules['gurobipy'] state (may not exist)."""
        self._had_gurobipy = "gurobipy" in sys.modules
        self._original = sys.modules.pop("gurobipy", None)

    def teardown_method(self):
        """Restore original sys.modules state."""
        if self._original is not None:
            sys.modules["gurobipy"] = self._original
        elif "gurobipy" in sys.modules:
            del sys.modules["gurobipy"]

    def test_install_patches_sys_modules(self):
        gp.install()
        assert sys.modules["gurobipy"] is gp

    def test_import_after_install_gets_wrapper(self):
        gp.install()
        module = importlib.import_module("gurobipy")
        assert module is gp

    def test_grb_accessible_via_patched_gurobipy(self):
        gp.install()
        module = importlib.import_module("gurobipy")
        assert module.GRB is gp.GRB

    def test_model_accessible_via_patched_gurobipy(self):
        gp.install()
        module = importlib.import_module("gurobipy")
        assert module.Model is gp.Model

    def test_quicksum_accessible_via_patched_gurobipy(self):
        gp.install()
        module = importlib.import_module("gurobipy")
        assert module.quicksum is gp.quicksum

    def test_install_is_idempotent(self):
        gp.install()
        gp.install()  # second call should not raise
        assert sys.modules["gurobipy"] is gp

    def test_model_created_via_patched_import_solves_correctly(self):
        gp.install()
        module = importlib.import_module("gurobipy")
        m = module.Model()
        x = m.addVar(lb=0.0)
        m.setObjective(x, module.GRB.MINIMIZE)
        m.addConstr(x >= 5)
        m.optimize()
        assert m.Status == module.GRB.OPTIMAL
        assert x.X == pytest.approx(5.0, abs=TOL)

    def test_install_callable_from_package(self):
        import grbcompat
        assert callable(grbcompat.install)

    def test_install_in_all(self):
        import grbcompat
        assert "install" in grbcompat.__all__


# ------------------------------------------------------------------ #
# GRB constants
# ------------------------------------------------------------------ #

class TestGRBConstants:
    def test_minimize(self):
        assert GRB.MINIMIZE == 1

    def test_maximize(self):
        assert GRB.MAXIMIZE == -1

    def test_continuous(self):
        assert GRB.CONTINUOUS == "C"

    def test_binary(self):
        assert GRB.BINARY == "B"

    def test_integer(self):
        assert GRB.INTEGER == "I"

    def test_less_equal(self):
        assert GRB.LESS_EQUAL == "<"

    def test_greater_equal(self):
        assert GRB.GREATER_EQUAL == ">"

    def test_equal(self):
        assert GRB.EQUAL == "="

    def test_optimal(self):
        assert GRB.OPTIMAL == 2

    def test_infeasible(self):
        assert GRB.INFEASIBLE == 3

    def test_status_class_mirrors_top_level(self):
        assert GRB.Status.OPTIMAL == GRB.OPTIMAL

    def test_grb_inf(self):
        assert GRB.INFINITY > 1e50

    def test_grb_attr(self):
        assert GRB.Attr.X == "X"
        assert GRB.Attr.Pi == "Pi"


# ------------------------------------------------------------------ #
# Exported names
# ------------------------------------------------------------------ #

class TestExportedNames:
    def test_GurobiError_importable(self):
        from grbcompat import GurobiError
        assert issubclass(GurobiError, Exception)

    def test_Env_importable(self):
        from grbcompat import Env
        e = Env()
        assert e is not None

    def test_disposeDefaultEnv_callable(self):
        gp.disposeDefaultEnv()

    def test_tupledict_importable(self):
        from grbcompat import tupledict
        td = tupledict()
        assert isinstance(td, dict)

    def test_LinExpr_importable(self):
        from grbcompat import LinExpr
        e = LinExpr()
        assert e.size() == 0


# ------------------------------------------------------------------ #
# Typical gurobipy usage patterns (using grbcompat directly)
# ------------------------------------------------------------------ #

class TestGurobiPyPatterns:
    def test_classic_lp(self):
        m = gp.Model("classic_lp")
        x = m.addVar(lb=0, name="x")
        y = m.addVar(lb=0, name="y")
        m.setObjective(x + 2 * y, GRB.MINIMIZE)
        m.addConstr(x + y >= 1, "c0")
        m.addConstr(x <= 3, "c1")
        m.optimize()
        assert m.Status == GRB.OPTIMAL
        assert m.ObjVal == pytest.approx(1.0, abs=TOL)
        assert x.X + y.X == pytest.approx(1.0, abs=TOL)

    def test_grb_constant_chaining(self):
        assert GRB.MINIMIZE == 1
        assert GRB.BINARY == "B"
        assert GRB.LESS_EQUAL == "<"

    def test_model_params_style(self):
        m = gp.Model()
        m.Params.OutputFlag = 0
        m.Params.TimeLimit = 60
        x = m.addVar(lb=0)
        m.setObjective(x, GRB.MINIMIZE)
        m.optimize()
        assert m.Status == GRB.OPTIMAL

    def test_model_setparam_style(self):
        m = gp.Model()
        m.setParam("OutputFlag", 0)
        m.setParam("TimeLimit", 60)
        x = m.addVar(lb=0)
        m.setObjective(x, GRB.MINIMIZE)
        m.optimize()
        assert m.Status == GRB.OPTIMAL

    def test_addvars_quicksum_pattern(self):
        n = 10
        m = gp.Model()
        x = m.addVars(n, vtype=GRB.BINARY, name="x")
        m.setObjective(gp.quicksum(x[i] for i in range(n)), GRB.MAXIMIZE)
        m.addConstr(gp.quicksum(x[i] for i in range(n)) <= 5)
        m.optimize()
        assert m.Status == GRB.OPTIMAL
        assert m.ObjVal == pytest.approx(5.0, abs=TOL)

    def test_addconstrs_generator_pattern(self):
        m = gp.Model()
        x = m.addVars(5, lb=0.0, name="x")
        m.setObjective(gp.quicksum(x[i] for i in range(5)), GRB.MINIMIZE)
        cs = m.addConstrs((x[i] >= i for i in range(5)), name="lb")
        m.optimize()
        assert m.Status == GRB.OPTIMAL
        assert m.ObjVal == pytest.approx(0 + 1 + 2 + 3 + 4, abs=TOL)
        assert len(cs) == 5

    def test_2d_transportation_pattern(self):
        supply = [20, 30]
        demand = [15, 20, 15]
        cost = [[2, 3, 1], [5, 4, 3]]
        m = gp.Model("transport")
        x = m.addVars(range(len(supply)), range(len(demand)), lb=0.0, name="flow")
        m.setObjective(
            gp.quicksum(cost[i][j] * x[i, j] for i in range(2) for j in range(3)),
            GRB.MINIMIZE,
        )
        for i in range(len(supply)):
            m.addConstr(gp.quicksum(x[i, j] for j in range(len(demand))) <= supply[i])
        for j in range(len(demand)):
            m.addConstr(gp.quicksum(x[i, j] for i in range(len(supply))) >= demand[j])
        m.optimize()
        assert m.Status == GRB.OPTIMAL

    def test_context_manager_pattern(self):
        with gp.Model("ctx") as m:
            x = m.addVar(lb=0.0)
            m.setObjective(x, GRB.MINIMIZE)
            m.addConstr(x >= 2)
            m.optimize()
            assert m.ObjVal == pytest.approx(2.0, abs=TOL)

    def test_getVarByName_pattern(self):
        m = gp.Model()
        x = m.addVar(lb=0.0, name="myvar")
        assert m.getVarByName("myvar") is x
        assert m.getVarByName("nope") is None

    def test_getConstrByName_pattern(self):
        m = gp.Model()
        x = m.addVar()
        c = m.addConstr(x <= 5, name="cap")
        assert m.getConstrByName("cap") is c

    def test_dual_retrieval_pattern(self):
        m = gp.Model()
        x = m.addVar(lb=0.0)
        c = m.addConstr(x >= 3, name="lb")
        m.setObjective(x, GRB.MINIMIZE)
        m.optimize()
        assert c.Pi == pytest.approx(1.0, abs=TOL)
        assert c.Slack == pytest.approx(0.0, abs=TOL)

    def test_env_context_manager_pattern(self):
        with gp.Env() as env:
            m = gp.Model("m", env=env)
            x = m.addVar()
            m.setObjective(x, GRB.MINIMIZE)
            m.optimize()
            assert m.Status == GRB.OPTIMAL


# ------------------------------------------------------------------ #
# quicksum
# ------------------------------------------------------------------ #

class TestQuicksum:
    def test_quicksum_returns_linexpr(self):
        from grbcompat import LinExpr
        m = gp.Model()
        x = m.addVars(3, name="x")
        result = gp.quicksum(x[i] for i in range(3))
        assert isinstance(result, LinExpr)

    def test_quicksum_correct_size(self):
        m = gp.Model()
        x = m.addVars(5, name="x")
        result = gp.quicksum(x[i] for i in range(5))
        assert result.size() == 5

    def test_quicksum_with_coefficients(self):
        m = gp.Model()
        x = m.addVars(3, name="x")
        coeffs = [1.0, 2.0, 3.0]
        result = gp.quicksum(coeffs[i] * x[i] for i in range(3))
        assert result.size() == 3

    def test_quicksum_empty_iterable(self):
        from grbcompat import LinExpr
        result = gp.quicksum([])
        assert isinstance(result, LinExpr)
        assert result.size() == 0

    def test_quicksum_single(self):
        m = gp.Model()
        x = m.addVar()
        result = gp.quicksum([x])
        assert result.size() == 1

    def test_quicksum_with_linexprs(self):
        m = gp.Model()
        x = m.addVars(3, name="x")
        exprs = [2 * x[i] + 1 for i in range(3)]
        result = gp.quicksum(exprs)
        assert result.size() == 3
        assert result.constant == pytest.approx(3.0)

    def test_quicksum_result_is_correct(self):
        m = gp.Model()
        x = m.addVar(lb=1.0, ub=1.0)
        y = m.addVar(lb=2.0, ub=2.0)
        z = m.addVar(lb=3.0, ub=3.0)
        m.addConstr(x == 1)
        m.addConstr(y == 2)
        m.addConstr(z == 3)
        m.setObjective(gp.quicksum([x, y, z]), GRB.MINIMIZE)
        m.optimize()
        assert m.ObjVal == pytest.approx(6.0, abs=TOL)


# ------------------------------------------------------------------ #
# multidict
# ------------------------------------------------------------------ #

class TestMultidict:
    def test_single_value(self):
        data = {0: 1.0, 1: 2.0, 2: 3.0}
        result = gp.multidict(data)
        assert result[0] == [0, 1, 2]
        assert result[1] == data

    def test_multi_value(self):
        data = {"a": [1, 10], "b": [2, 20], "c": [3, 30]}
        keys, d1, d2 = gp.multidict(data)
        assert set(keys) == {"a", "b", "c"}
        assert d1["a"] == 1
        assert d2["a"] == 10

    def test_empty(self):
        result = gp.multidict({})
        assert result == [[]]


# ------------------------------------------------------------------ #
# Env stub
# ------------------------------------------------------------------ #

class TestEnvStub:
    def test_env_instantiation(self):
        e = gp.Env()
        assert e is not None

    def test_env_start(self):
        gp.Env().start()

    def test_env_dispose(self):
        gp.Env().dispose()

    def test_env_context_manager(self):
        with gp.Env() as e:
            assert e is not None


# ------------------------------------------------------------------ #
# LinExpr via wrapper
# ------------------------------------------------------------------ #

class TestLinExprViaPatch:
    def test_linexpr_addition(self):
        from grbcompat import LinExpr
        m = gp.Model()
        x = m.addVar()
        y = m.addVar()
        e = x + y
        assert isinstance(e, LinExpr)

    def test_linexpr_comparison(self):
        from grbcompat._expr import TempConstr
        m = gp.Model()
        x = m.addVar()
        tc = x <= 5
        assert isinstance(tc, TempConstr)

    def test_sum_builtin_on_vars(self):
        from grbcompat import LinExpr
        m = gp.Model()
        vars_ = [m.addVar() for _ in range(4)]
        e = sum(vars_)
        assert isinstance(e, LinExpr)
        assert e.size() == 4

    def test_linexpr_used_in_constr(self):
        m = gp.Model()
        x = m.addVar(lb=0.0)
        y = m.addVar(lb=0.0)
        e = 2 * x + 3 * y
        c = m.addConstr(e <= 12)
        m.setObjective(e, GRB.MAXIMIZE)
        m.optimize()
        assert m.Status == GRB.OPTIMAL
        assert m.ObjVal == pytest.approx(12.0, abs=TOL)


# ------------------------------------------------------------------ #
# Status codes via wrapper
# ------------------------------------------------------------------ #

class TestStatusCodes:
    def test_grb_optimal(self):
        assert GRB.OPTIMAL == 2

    def test_grb_status_optimal(self):
        assert GRB.Status.OPTIMAL == 2

    def test_infeasible_model_status(self):
        m = gp.Model()
        x = m.addVar(lb=0.0)
        m.addConstr(x >= 5)
        m.addConstr(x <= 3)
        m.setObjective(x, GRB.MINIMIZE)
        m.optimize()
        assert m.Status == GRB.INFEASIBLE

    def test_optimal_model_status(self):
        m = gp.Model()
        x = m.addVar(lb=0.0)
        m.setObjective(x, GRB.MINIMIZE)
        m.optimize()
        assert m.Status == GRB.OPTIMAL
