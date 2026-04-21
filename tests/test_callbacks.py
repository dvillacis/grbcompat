"""Tests for model.optimize(callback), cbGet, cbGetSolution, cbLazy, terminate."""

from __future__ import annotations

import math
import pytest

import grbcompat as gp
from grbcompat import GRB


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def fresh_model():
    m = gp.Model()
    m.Params.OutputFlag = 0
    return m


def simple_binary_mip():
    """max x0 + 2*x1 + 3*x2, x in {0,1}^3, x0+x1+x2 <= 2"""
    m = fresh_model()
    x = [m.addVar(vtype=GRB.BINARY, name=f"x{i}") for i in range(3)]
    m.addConstr(x[0] + x[1] + x[2] <= 2)
    m.setObjective(x[0] + 2 * x[1] + 3 * x[2], GRB.MAXIMIZE)
    return m, x


def knapsack_no_capacity():
    """max x0 + 2*x1, x0,x1 in {0..5} — no capacity constraint initially."""
    m = fresh_model()
    x0 = m.addVar(lb=0, ub=5, vtype=GRB.INTEGER, name="x0")
    x1 = m.addVar(lb=0, ub=5, vtype=GRB.INTEGER, name="x1")
    m.setObjective(x0 + 2 * x1, GRB.MAXIMIZE)
    return m, x0, x1


# ------------------------------------------------------------------ #
# TestCallbackConstants
# ------------------------------------------------------------------ #

class TestCallbackConstants:

    def test_mipsol_where_code(self):
        assert GRB.Callback.MIPSOL == 8

    def test_mipnode_where_code(self):
        assert GRB.Callback.MIPNODE == 9

    def test_mip_where_code(self):
        assert GRB.Callback.MIP == 7

    def test_simplex_where_code(self):
        assert GRB.Callback.SIMPLEX == 6

    def test_barrier_where_code(self):
        assert GRB.Callback.BARRIER == 11

    def test_mipsol_what_codes_are_distinct(self):
        codes = [
            GRB.Callback.MIPSOL_OBJ,
            GRB.Callback.MIPSOL_OBJBST,
            GRB.Callback.MIPSOL_OBJBND,
            GRB.Callback.MIPSOL_NODCNT,
            GRB.Callback.MIPSOL_SOLCNT,
        ]
        assert len(set(codes)) == len(codes)

    def test_what_codes_are_ints(self):
        assert isinstance(GRB.Callback.MIPSOL_OBJ, int)
        assert isinstance(GRB.Callback.MIPSOL_SOLCNT, int)


# ------------------------------------------------------------------ #
# TestOptimizeNoCallback
# ------------------------------------------------------------------ #

class TestOptimizeNoCallback:

    def test_no_callback_lp(self):
        m, x = simple_binary_mip()
        m.optimize()
        assert m.Status == GRB.OPTIMAL

    def test_no_callback_mip(self):
        m = fresh_model()
        x = m.addVar(lb=0, ub=10, vtype=GRB.INTEGER)
        m.setObjective(x, GRB.MAXIMIZE)
        m.optimize()
        assert m.Status == GRB.OPTIMAL
        assert math.isclose(x.X, 10.0, abs_tol=1e-5)

    def test_optimize_none_explicit(self):
        m, x = simple_binary_mip()
        m.optimize(callback=None)
        assert m.Status == GRB.OPTIMAL


# ------------------------------------------------------------------ #
# TestMIPSOLCallback
# ------------------------------------------------------------------ #

class TestMIPSOLCallback:

    def test_callback_fires_for_mip(self):
        m, x = simple_binary_mip()
        fired = []
        def cb(where):
            fired.append(where)
        m.optimize(cb)
        assert len(fired) >= 1

    def test_callback_where_is_mipsol(self):
        m, x = simple_binary_mip()
        where_vals = []
        def cb(where):
            where_vals.append(where)
        m.optimize(cb)
        assert all(w == GRB.Callback.MIPSOL for w in where_vals)

    def test_callback_not_fired_for_lp(self):
        m = fresh_model()
        x = m.addVar(lb=0.0, ub=1.0)   # continuous
        m.setObjective(x, GRB.MAXIMIZE)
        fired = []
        def cb(where):
            fired.append(where)
        m.optimize(cb)
        assert len(fired) == 0

    def test_cbgetsolution_single_var(self):
        m, x = simple_binary_mip()
        vals = []
        def cb(where):
            vals.append(m.cbGetSolution(x[0]))
        m.optimize(cb)
        assert len(vals) >= 1
        assert isinstance(vals[0], float)

    def test_cbgetsolution_list_of_vars(self):
        m, x = simple_binary_mip()
        results = []
        def cb(where):
            results.append(m.cbGetSolution(x))
        m.optimize(cb)
        assert len(results) >= 1
        assert isinstance(results[0], list)
        assert len(results[0]) == 3

    def test_cbgetsolution_tupledict(self):
        m = fresh_model()
        x = m.addVars(3, vtype=GRB.BINARY, name="x")
        m.addConstr(x[0] + x[1] + x[2] <= 2)
        m.setObjective(x[0] + 2 * x[1] + 3 * x[2], GRB.MAXIMIZE)
        results = []
        def cb(where):
            d = m.cbGetSolution(x)
            results.append(d)
        m.optimize(cb)
        assert len(results) >= 1
        assert isinstance(results[0], dict)
        assert set(results[0].keys()) == {0, 1, 2}

    def test_cbget_mipsol_obj(self):
        m, x = simple_binary_mip()
        objs = []
        def cb(where):
            objs.append(m.cbGet(GRB.Callback.MIPSOL_OBJ))
        m.optimize(cb)
        assert len(objs) >= 1
        assert isinstance(objs[0], float)

    def test_cbget_mipsol_objbst(self):
        m, x = simple_binary_mip()
        vals = []
        def cb(where):
            vals.append(m.cbGet(GRB.Callback.MIPSOL_OBJBST))
        m.optimize(cb)
        assert isinstance(vals[0], float)

    def test_cbget_mipsol_objbnd(self):
        m, x = simple_binary_mip()
        vals = []
        def cb(where):
            vals.append(m.cbGet(GRB.Callback.MIPSOL_OBJBND))
        m.optimize(cb)
        assert isinstance(vals[0], float)

    def test_cbget_mipsol_nodcnt(self):
        m, x = simple_binary_mip()
        vals = []
        def cb(where):
            vals.append(m.cbGet(GRB.Callback.MIPSOL_NODCNT))
        m.optimize(cb)
        assert isinstance(vals[0], int)
        assert vals[0] >= 0

    def test_cbget_mipsol_solcnt(self):
        m, x = simple_binary_mip()
        counts = []
        def cb(where):
            counts.append(m.cbGet(GRB.Callback.MIPSOL_SOLCNT))
        m.optimize(cb)
        assert len(counts) >= 1
        assert counts[0] >= 1

    def test_solcnt_increments(self):
        """On problems with multiple feasible solutions, SOLCNT increases."""
        m = fresh_model()
        # Large enough that multiple solutions might be found
        x = [m.addVar(lb=0, ub=5, vtype=GRB.INTEGER) for _ in range(3)]
        m.addConstr(x[0] + x[1] + x[2] <= 10)
        m.setObjective(x[0] + 2 * x[1] + 3 * x[2], GRB.MAXIMIZE)
        counts = []
        def cb(where):
            counts.append(m.cbGet(GRB.Callback.MIPSOL_SOLCNT))
        m.optimize(cb)
        # At least one solution was found
        assert len(counts) >= 1
        assert counts[-1] == len(counts)

    def test_solution_in_callback_is_feasible(self):
        """Solution values in callback must satisfy the capacity constraint."""
        m, x = simple_binary_mip()
        violations = []
        def cb(where):
            vals = m.cbGetSolution(x)
            s = sum(vals)
            if s > 2.0 + 1e-6:
                violations.append(s)
        m.optimize(cb)
        assert violations == [], f"Constraint violated in callback: {violations}"

    def test_callback_solution_is_final_solution(self):
        """For simple problems the callback solution matches the final solution."""
        m, x = simple_binary_mip()
        cb_sols = []
        def cb(where):
            cb_sols.append(m.cbGetSolution(x))
        m.optimize(cb)
        # Last callback solution should match final optimum
        assert m.Status == GRB.OPTIMAL
        final = [v.X for v in x]
        last_cb = cb_sols[-1]
        assert all(math.isclose(a, b, abs_tol=1e-5) for a, b in zip(last_cb, final))


# ------------------------------------------------------------------ #
# TestTerminate
# ------------------------------------------------------------------ #

class TestTerminate:

    def test_terminate_stops_solve(self):
        """Calling terminate() inside MIPSOL stops the solver."""
        m = fresh_model()
        x = [m.addVar(lb=0, ub=10, vtype=GRB.INTEGER) for _ in range(4)]
        m.addConstr(gp.quicksum(x) <= 30)
        m.setObjective(gp.quicksum((i + 1) * x[i] for i in range(4)), GRB.MAXIMIZE)

        calls = [0]
        def cb(where):
            calls[0] += 1
            if calls[0] >= 1:
                m.terminate()

        m.optimize(cb)
        # Solver terminated — still has a solution
        assert m._solution is not None
        assert m.Status in (GRB.OPTIMAL, GRB.INTERRUPTED, GRB.TIME_LIMIT, GRB.SOLUTION_LIMIT)

    def test_terminate_solution_accessible(self):
        """After terminate(), solution values are still accessible."""
        m, x = simple_binary_mip()
        def cb(where):
            m.terminate()
        m.optimize(cb)
        # Variable values should be accessible
        for v in x:
            _ = v.X  # should not raise

    def test_terminate_outside_callback_is_noop(self):
        """terminate() outside a callback should not raise."""
        m = fresh_model()
        m.terminate()  # no-op, no error

    def test_terminate_outside_optimize_is_noop(self):
        """Calling terminate() before optimize() should not crash optimize()."""
        m, x = simple_binary_mip()
        m.terminate()  # sets nothing since _cb_data_in is None
        fired = []
        def cb(where):
            fired.append(where)
        m.optimize(cb)
        assert len(fired) >= 1  # callback still fires


# ------------------------------------------------------------------ #
# TestCbGet
# ------------------------------------------------------------------ #

class TestCbGet:

    def test_cbget_outside_callback_raises(self):
        m = fresh_model()
        with pytest.raises(RuntimeError, match="outside of a callback"):
            m.cbGet(GRB.Callback.MIPSOL_OBJ)

    def test_cbgetsolution_outside_callback_raises(self):
        m = fresh_model()
        x = m.addVar()
        with pytest.raises(RuntimeError, match="outside of a callback"):
            m.cbGetSolution(x)

    def test_cbget_unknown_code_raises(self):
        m, x = simple_binary_mip()
        errors = []
        def cb(where):
            try:
                m.cbGet(99999)
            except AttributeError:
                errors.append(True)
        m.optimize(cb)
        assert errors  # at least one AttributeError was raised

    def test_cbget_after_optimize_raises(self):
        m, x = simple_binary_mip()
        m.optimize()
        with pytest.raises(RuntimeError, match="outside of a callback"):
            m.cbGet(GRB.Callback.MIPSOL_OBJ)


# ------------------------------------------------------------------ #
# TestCbLazy
# ------------------------------------------------------------------ #

class TestCbLazy:

    def test_cblazy_outside_callback_raises(self):
        m, x0, x1 = knapsack_no_capacity()
        with pytest.raises(RuntimeError, match="MIPSOL callback"):
            m.cbLazy(x0 + x1 <= 6)

    def test_cblazy_adds_constraint_before_resolving(self):
        """Lazy constraint is added to model after first solve."""
        m, x0, x1 = knapsack_no_capacity()
        initial_constrs = m.NumConstrs

        def cb(where):
            sol = m.cbGetSolution([x0, x1])
            if sol[0] + sol[1] > 6.5:
                m.cbLazy(x0 + x1 <= 6)

        m.optimize(cb)
        # Lazy constraint was added to the model
        assert m.NumConstrs > initial_constrs

    def test_cblazy_outer_loop_correct_solution(self):
        """
        Problem with no capacity constraint initially.
        Lazy: x0 + x1 <= 6.  Expected solution: x0=1, x1=5 (obj=11).
        Without lazy: x0=5, x1=5 (obj=15, sum=10 > 6).
        """
        m, x0, x1 = knapsack_no_capacity()

        def cb(where):
            sol = m.cbGetSolution([x0, x1])
            if sol[0] + sol[1] > 6.5:
                m.cbLazy(x0 + x1 <= 6)

        m.optimize(cb)
        assert m.Status == GRB.OPTIMAL
        assert x0.X + x1.X <= 6.0 + 1e-5

    def test_cblazy_multiple_constraints_in_one_callback(self):
        """Multiple cbLazy calls in a single callback invocation are all queued."""
        m = fresh_model()
        x0 = m.addVar(lb=0, ub=10, vtype=GRB.INTEGER)
        x1 = m.addVar(lb=0, ub=10, vtype=GRB.INTEGER)
        x2 = m.addVar(lb=0, ub=10, vtype=GRB.INTEGER)
        m.setObjective(x0 + x1 + x2, GRB.MAXIMIZE)
        initial_constrs = m.NumConstrs

        called = [False]
        def cb(where):
            if not called[0]:
                called[0] = True
                m.cbLazy(x0 <= 5)
                m.cbLazy(x1 <= 5)
                m.cbLazy(x2 <= 5)

        m.optimize(cb)
        # All three lazy constraints were added
        assert m.NumConstrs >= initial_constrs + 3

    def test_cblazy_constraints_appear_in_numconstrs(self):
        m, x0, x1 = knapsack_no_capacity()
        initial_constrs = m.NumConstrs

        def cb(where):
            sol = m.cbGetSolution([x0, x1])
            if sol[0] + sol[1] > 6.5:
                m.cbLazy(x0 + x1 <= 6)

        m.optimize(cb)
        assert m.NumConstrs == initial_constrs + 1

    def test_params_lazy_constraints_noop(self):
        m = fresh_model()
        m.Params.LazyConstraints = 1  # should not raise
        m.Params.LazyConstraints = 0
