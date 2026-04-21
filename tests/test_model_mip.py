"""MIP solve tests – binary and integer programming."""

import pytest
from grbcompat import GRB, Model, quicksum


TOL = 1e-6
INT_TOL = 1e-5


def is_integer(v: float, tol: float = INT_TOL) -> bool:
    return abs(v - round(v)) <= tol


# ------------------------------------------------------------------ #
# Binary programs
# ------------------------------------------------------------------ #

class TestBinaryPrograms:
    def test_single_binary_maximize(self):
        """max x  s.t. x in {0,1}  →  x*=1, obj=1."""
        m = Model()
        x = m.addVar(vtype=GRB.BINARY, name="x")
        m.setObjective(x, GRB.MAXIMIZE)
        m.optimize()
        assert m.Status == GRB.OPTIMAL
        assert m.ObjVal == pytest.approx(1.0, abs=TOL)
        assert x.X == pytest.approx(1.0, abs=TOL)

    def test_single_binary_minimize(self):
        """min x  s.t. x in {0,1}  →  x*=0, obj=0."""
        m = Model()
        x = m.addVar(vtype=GRB.BINARY, name="x")
        m.setObjective(x, GRB.MINIMIZE)
        m.optimize()
        assert m.Status == GRB.OPTIMAL
        assert m.ObjVal == pytest.approx(0.0, abs=TOL)

    def test_binary_forced_to_one(self):
        """min x  s.t. x in {0,1}, x >= 0.5  →  x*=1."""
        m = Model()
        x = m.addVar(vtype=GRB.BINARY)
        m.setObjective(x, GRB.MINIMIZE)
        m.addConstr(x >= 0.5)
        m.optimize()
        assert m.Status == GRB.OPTIMAL
        assert x.X == pytest.approx(1.0, abs=TOL)

    def test_knapsack_01(self):
        """
        0-1 knapsack
        values  = [10, 6, 4, 5, 3]
        weights = [ 5, 4, 3, 2, 1]
        capacity = 8
        Optimal items: {0, 3, 4}, obj = 18.
        """
        values  = [10, 6, 4, 5, 3]
        weights = [ 5, 4, 3, 2, 1]
        capacity = 8
        n = len(values)
        m = Model()
        x = m.addVars(n, vtype=GRB.BINARY, name="x")
        m.setObjective(quicksum(values[i] * x[i] for i in range(n)), GRB.MAXIMIZE)
        m.addConstr(quicksum(weights[i] * x[i] for i in range(n)) <= capacity, "cap")
        m.optimize()
        assert m.Status == GRB.OPTIMAL
        assert m.ObjVal == pytest.approx(18.0, abs=TOL)
        # Verify integrality
        assert all(is_integer(x[i].X) for i in range(n))
        # Verify capacity
        assert sum(weights[i] * x[i].X for i in range(n)) <= capacity + TOL

    def test_binary_with_multiple_constraints(self):
        """
        max x0 + x1 + x2
        s.t. x0 + x1 <= 1  (at most one of 0,1)
             x1 + x2 <= 1  (at most one of 1,2)
             xi in {0,1}
        Optimal: x0=1, x1=0, x2=1, obj=2.
        """
        m = Model()
        x = m.addVars(3, vtype=GRB.BINARY, name="x")
        m.setObjective(quicksum(x[i] for i in range(3)), GRB.MAXIMIZE)
        m.addConstr(x[0] + x[1] <= 1)
        m.addConstr(x[1] + x[2] <= 1)
        m.optimize()
        assert m.Status == GRB.OPTIMAL
        assert m.ObjVal == pytest.approx(2.0, abs=TOL)

    def test_infeasible_binary(self):
        """x in {0,1} but x >= 2 → infeasible."""
        m = Model()
        x = m.addVar(vtype=GRB.BINARY)
        m.addConstr(x >= 2)
        m.setObjective(x, GRB.MINIMIZE)
        m.optimize()
        assert m.Status == GRB.INFEASIBLE


# ------------------------------------------------------------------ #
# Integer programs
# ------------------------------------------------------------------ #

class TestIntegerPrograms:
    def test_single_integer_minimize(self):
        """min x  s.t. x >= 2.5, x integer  →  x*=3, obj=3."""
        m = Model()
        x = m.addVar(lb=0.0, vtype=GRB.INTEGER)
        m.setObjective(x, GRB.MINIMIZE)
        m.addConstr(x >= 2.5)
        m.optimize()
        assert m.Status == GRB.OPTIMAL
        assert x.X == pytest.approx(3.0, abs=TOL)
        assert is_integer(x.X)

    def test_single_integer_maximize(self):
        """max x  s.t. x <= 7.8, x integer  →  x*=7, obj=7."""
        m = Model()
        x = m.addVar(lb=0.0, vtype=GRB.INTEGER)
        m.setObjective(x, GRB.MAXIMIZE)
        m.addConstr(x <= 7.8)
        m.optimize()
        assert m.Status == GRB.OPTIMAL
        assert x.X == pytest.approx(7.0, abs=TOL)

    def test_integer_integrality(self):
        m = Model()
        x = m.addVars(5, lb=0.0, vtype=GRB.INTEGER, name="x")
        m.setObjective(quicksum(x[i] for i in range(5)), GRB.MINIMIZE)
        for i in range(5):
            m.addConstr(x[i] >= i + 0.7)
        m.optimize()
        assert m.Status == GRB.OPTIMAL
        for i in range(5):
            assert is_integer(x[i].X)

    def test_infeasible_integer(self):
        """x integer, x >= 5, x <= 3 → infeasible."""
        m = Model()
        x = m.addVar(lb=0.0, vtype=GRB.INTEGER)
        m.addConstr(x >= 5)
        m.addConstr(x <= 3)
        m.setObjective(x, GRB.MINIMIZE)
        m.optimize()
        assert m.Status == GRB.INFEASIBLE


# ------------------------------------------------------------------ #
# Mixed-integer programs
# ------------------------------------------------------------------ #

class TestMixedIntegerPrograms:
    def test_mixed_int_cont(self):
        """
        min x + y
        s.t. x + y >= 3
             x integer, y continuous, both >= 0
        LP relaxation optimal: x=0, y=3.
        MIP optimal: x=0, y=3 (same here since y is cont).
        """
        m = Model()
        x = m.addVar(lb=0.0, vtype=GRB.INTEGER, name="x")
        y = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="y")
        m.setObjective(x + y, GRB.MINIMIZE)
        m.addConstr(x + y >= 3)
        m.optimize()
        assert m.Status == GRB.OPTIMAL
        assert m.ObjVal == pytest.approx(3.0, abs=TOL)
        assert is_integer(x.X)

    def test_ip_objective_worse_than_lp_relaxation(self):
        """
        min x + y
        s.t. x + y >= 2.5, x,y integer, x,y >= 0
        LP rel: obj=2.5; MIP: obj=3 (x=1,y=2 or x=2,y=1 or x=3,y=0 etc.)
        """
        m = Model()
        x = m.addVar(lb=0.0, vtype=GRB.INTEGER)
        y = m.addVar(lb=0.0, vtype=GRB.INTEGER)
        m.setObjective(x + y, GRB.MINIMIZE)
        m.addConstr(x + y >= 2.5)
        m.optimize()
        assert m.Status == GRB.OPTIMAL
        assert m.ObjVal == pytest.approx(3.0, abs=TOL)
        assert is_integer(x.X) and is_integer(y.X)

    def test_set_cover(self):
        """
        Minimal set cover:
        Universe = {0,1,2,3,4}
        Sets: S0={0,1,2}, S1={1,3,4}, S2={0,2,4}, S3={3}
        Cover with fewest sets.
        Optimal: S0 ∪ S1  (or similar), obj=2.
        """
        sets = [
            {0, 1, 2},
            {1, 3, 4},
            {0, 2, 4},
            {3},
        ]
        universe = range(5)
        m = Model()
        x = m.addVars(len(sets), vtype=GRB.BINARY, name="x")
        m.setObjective(quicksum(x[s] for s in range(len(sets))), GRB.MINIMIZE)
        for e in universe:
            m.addConstr(quicksum(x[s] for s in range(len(sets)) if e in sets[s]) >= 1)
        m.optimize()
        assert m.Status == GRB.OPTIMAL
        assert m.ObjVal == pytest.approx(2.0, abs=TOL)
        # Verify coverage
        for e in universe:
            assert sum(x[s].X for s in range(len(sets)) if e in sets[s]) >= 1 - TOL


# ------------------------------------------------------------------ #
# Verify variable type after solve
# ------------------------------------------------------------------ #

class TestMIPIntegrality:
    def test_binary_solution_is_01(self):
        m = Model()
        x = m.addVars(10, vtype=GRB.BINARY, name="x")
        m.setObjective(quicksum((i % 3) * x[i] for i in range(10)), GRB.MAXIMIZE)
        for i in range(0, 10, 2):
            m.addConstr(x[i] + x[i + 1] <= 1)
        m.optimize()
        assert m.Status == GRB.OPTIMAL
        for i in range(10):
            assert x[i].X == pytest.approx(0.0, abs=TOL) or x[i].X == pytest.approx(1.0, abs=TOL)

    def test_integer_solution_is_integral(self):
        m = Model()
        x = m.addVars(6, lb=0.0, vtype=GRB.INTEGER, name="x")
        m.setObjective(quicksum(x[i] for i in range(6)), GRB.MINIMIZE)
        m.addConstr(quicksum(x[i] for i in range(6)) >= 10)
        for i in range(6):
            m.addConstr(x[i] >= i * 0.3)
        m.optimize()
        assert m.Status == GRB.OPTIMAL
        for i in range(6):
            assert is_integer(x[i].X)
