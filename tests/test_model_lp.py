"""LP solve tests - correctness of objective, solution values, and duals."""

import pytest
from grbcompat import GRB, Model


TOL = 1e-6


# ------------------------------------------------------------------ #
# Basic feasibility and optimality
# ------------------------------------------------------------------ #

class TestLPStatus:
    def test_optimal_status(self):
        m = Model()
        x = m.addVar(lb=0.0)
        m.setObjective(x, GRB.MINIMIZE)
        m.addConstr(x >= 1)
        m.optimize()
        assert m.Status == GRB.OPTIMAL

    def test_infeasible_status(self):
        m = Model()
        x = m.addVar(lb=0.0)
        m.addConstr(x >= 5)
        m.addConstr(x <= 3)
        m.setObjective(x, GRB.MINIMIZE)
        m.optimize()
        assert m.Status == GRB.INFEASIBLE

    def test_unbounded_status(self):
        m = Model()
        x = m.addVar(lb=-GRB.INFINITY)
        m.setObjective(x, GRB.MINIMIZE)  # minimize x with no lower bound
        m.optimize()
        assert m.Status in (GRB.UNBOUNDED, GRB.INF_OR_UNBD)

    def test_status_is_integer(self):
        m = Model()
        x = m.addVar()
        m.setObjective(x, GRB.MINIMIZE)
        m.optimize()
        assert isinstance(m.Status, int)


# ------------------------------------------------------------------ #
# Minimize: objective value
# ------------------------------------------------------------------ #

class TestLPMinimize:
    def test_single_var_minimize(self):
        """min x  s.t. x >= 3  →  x*=3, obj=3."""
        m = Model()
        x = m.addVar(lb=0.0)
        m.setObjective(x, GRB.MINIMIZE)
        m.addConstr(x >= 3)
        m.optimize()
        assert m.ObjVal == pytest.approx(3.0, abs=TOL)
        assert x.X == pytest.approx(3.0, abs=TOL)

    def test_two_var_minimize(self):
        """min x+2y  s.t. x+y>=1, x,y>=0  →  x=1,y=0, obj=1."""
        m = Model()
        x = m.addVar(lb=0.0, name="x")
        y = m.addVar(lb=0.0, name="y")
        m.setObjective(x + 2 * y, GRB.MINIMIZE)
        m.addConstr(x + y >= 1)
        m.optimize()
        assert m.ObjVal == pytest.approx(1.0, abs=TOL)

    def test_minimize_with_equality_constraint(self):
        """min x+y  s.t. x+y==5, x,y>=0  →  obj=5."""
        m = Model()
        x = m.addVar(lb=0.0)
        y = m.addVar(lb=0.0)
        m.setObjective(x + y, GRB.MINIMIZE)
        m.addConstr(x + y == 5)
        m.optimize()
        assert m.ObjVal == pytest.approx(5.0, abs=TOL)
        assert x.X + y.X == pytest.approx(5.0, abs=TOL)

    def test_minimize_with_upper_bound(self):
        """min -x  s.t. x<=4  →  x*=4, obj=-4."""
        m = Model()
        x = m.addVar(lb=0.0, ub=4.0)
        m.setObjective(-x, GRB.MINIMIZE)
        m.optimize()
        assert m.ObjVal == pytest.approx(-4.0, abs=TOL)
        assert x.X == pytest.approx(4.0, abs=TOL)

    def test_minimize_with_negative_lb(self):
        """min x  s.t. x>=-5, x<=0  →  x*=-5, obj=-5."""
        m = Model()
        x = m.addVar(lb=-5.0, ub=0.0)
        m.setObjective(x, GRB.MINIMIZE)
        m.optimize()
        assert m.ObjVal == pytest.approx(-5.0, abs=TOL)

    def test_minimize_constant_objective(self):
        """min 0  →  status OPTIMAL, ObjVal=0."""
        m = Model()
        x = m.addVar(lb=0.0, ub=1.0)
        from grbcompat._expr import LinExpr
        m.setObjective(LinExpr(), GRB.MINIMIZE)
        m.optimize()
        assert m.Status == GRB.OPTIMAL
        assert m.ObjVal == pytest.approx(0.0, abs=TOL)

    def test_multiple_constraints(self):
        """
        min 3x + 2y
        s.t. x + y  >= 4
             x      <= 3
             y      >= 1
        Since cost(y)=2 < cost(x)=3, optimal uses y as much as possible:
        x=0, y=4, obj=8.
        """
        m = Model()
        x = m.addVar(lb=0.0)
        y = m.addVar(lb=0.0)
        m.setObjective(3 * x + 2 * y, GRB.MINIMIZE)
        m.addConstr(x + y >= 4)
        m.addConstr(x <= 3)
        m.addConstr(y >= 1)
        m.optimize()
        assert m.ObjVal == pytest.approx(8.0, abs=TOL)
        assert x.X == pytest.approx(0.0, abs=TOL)
        assert y.X == pytest.approx(4.0, abs=TOL)


# ------------------------------------------------------------------ #
# Maximize: objective value
# ------------------------------------------------------------------ #

class TestLPMaximize:
    def test_single_var_maximize(self):
        """max x  s.t. x<=6  →  x*=6, obj=6."""
        m = Model()
        x = m.addVar(lb=0.0, ub=6.0)
        m.setObjective(x, GRB.MAXIMIZE)
        m.optimize()
        assert m.ObjVal == pytest.approx(6.0, abs=TOL)
        assert x.X == pytest.approx(6.0, abs=TOL)

    def test_two_var_maximize(self):
        """
        max x + y
        s.t. x + y <= 5, x <= 3, y <= 4, x,y >= 0
        Optimal: obj=5.
        """
        m = Model()
        x = m.addVar(lb=0.0)
        y = m.addVar(lb=0.0)
        m.setObjective(x + y, GRB.MAXIMIZE)
        m.addConstr(x + y <= 5)
        m.addConstr(x <= 3)
        m.addConstr(y <= 4)
        m.optimize()
        assert m.ObjVal == pytest.approx(5.0, abs=TOL)

    def test_maximize_with_equality(self):
        """max x  s.t. x+y==3, y>=0  →  y=0, x=3, obj=3."""
        m = Model()
        x = m.addVar(lb=0.0)
        y = m.addVar(lb=0.0)
        m.setObjective(x, GRB.MAXIMIZE)
        m.addConstr(x + y == 3)
        m.optimize()
        assert m.ObjVal == pytest.approx(3.0, abs=TOL)


# ------------------------------------------------------------------ #
# Dual values and slacks
# ------------------------------------------------------------------ #

class TestLPDualsAndSlacks:
    """
    minimize  2x + 3y
    s.t.      x + y >= 4   (binding, Pi = shadow price)
              x     <= 6   (slack = 2)
              x, y  >= 0
    Optimal: x=4, y=0, obj=8.
    """

    def setup_method(self):
        self.m = Model()
        x = self.m.addVar(lb=0.0, name="x")
        y = self.m.addVar(lb=0.0, name="y")
        self.c0 = self.m.addConstr(x + y >= 4, name="demand")
        self.c1 = self.m.addConstr(x <= 6, name="cap")
        self.m.setObjective(2 * x + 3 * y, GRB.MINIMIZE)
        self.m.optimize()
        self.x, self.y = x, y

    def test_optimal_obj(self):
        assert self.m.ObjVal == pytest.approx(8.0, abs=TOL)

    def test_x_value(self):
        assert self.x.X == pytest.approx(4.0, abs=TOL)

    def test_y_value(self):
        assert self.y.X == pytest.approx(0.0, abs=TOL)

    def test_binding_constraint_slack_zero(self):
        assert self.c0.Slack == pytest.approx(0.0, abs=TOL)

    def test_nonbinding_constraint_slack_positive(self):
        # x<=6, x=4 → slack = 2
        assert self.c1.Slack == pytest.approx(2.0, abs=TOL)

    def test_binding_constraint_pi_nonzero(self):
        assert abs(self.c0.Pi) > 1e-6

    def test_nonbinding_constraint_pi_zero(self):
        assert self.c1.Pi == pytest.approx(0.0, abs=TOL)

    def test_shadow_price_is_objective_cost(self):
        # For min 2x+3y, x+y>=4, cheapest way to satisfy is using x (cost 2)
        # Shadow price of demand >= 4 should equal 2
        assert self.c0.Pi == pytest.approx(2.0, abs=TOL)


# ------------------------------------------------------------------ #
# Reduced costs
# ------------------------------------------------------------------ #

class TestLPReducedCosts:
    def test_basic_variable_rc(self):
        """min x+2y  s.t. x+y>=1, x,y>=0.  x is in basis, y is not → RC_y>0."""
        m = Model()
        x = m.addVar(lb=0.0, name="x")
        y = m.addVar(lb=0.0, name="y")
        m.setObjective(x + 2 * y, GRB.MINIMIZE)
        m.addConstr(x + y >= 1)
        m.optimize()
        assert isinstance(x.RC, float)
        assert isinstance(y.RC, float)


# ------------------------------------------------------------------ #
# Runtime
# ------------------------------------------------------------------ #

class TestLPRuntime:
    def test_runtime_non_negative(self):
        m = Model()
        x = m.addVar(lb=0.0)
        m.setObjective(x, GRB.MINIMIZE)
        m.optimize()
        assert m.Runtime >= 0.0

    def test_runtime_is_float(self):
        m = Model()
        x = m.addVar(lb=0.0)
        m.setObjective(x, GRB.MINIMIZE)
        m.optimize()
        assert isinstance(m.Runtime, float)


# ------------------------------------------------------------------ #
# Objective constant
# ------------------------------------------------------------------ #

class TestLPObjectiveConstant:
    def test_obj_constant_appears_in_obj_val(self):
        """min x + 7  s.t. x>=2  →  ObjVal = 9."""
        m = Model()
        x = m.addVar(lb=0.0)
        m.setObjective(x + 7, GRB.MINIMIZE)
        m.addConstr(x >= 2)
        m.optimize()
        assert m.ObjVal == pytest.approx(9.0, abs=TOL)

    def test_negative_obj_constant(self):
        """min x - 3  s.t. x>=5  →  ObjVal = 2."""
        m = Model()
        x = m.addVar(lb=0.0)
        m.setObjective(x - 3, GRB.MINIMIZE)
        m.addConstr(x >= 5)
        m.optimize()
        assert m.ObjVal == pytest.approx(2.0, abs=TOL)


# ------------------------------------------------------------------ #
# Re-optimization after model changes
# ------------------------------------------------------------------ #

class TestLPReoptimize:
    def test_reoptimize_after_rhs_change(self):
        m = Model()
        x = m.addVar(lb=0.0)
        c = m.addConstr(x >= 3)
        m.setObjective(x, GRB.MINIMIZE)
        m.optimize()
        assert x.X == pytest.approx(3.0, abs=TOL)
        c.RHS = 7.0
        m.optimize()
        assert x.X == pytest.approx(7.0, abs=TOL)

    def test_reoptimize_after_obj_change(self):
        m = Model()
        x = m.addVar(lb=0.0, ub=5.0)
        y = m.addVar(lb=0.0, ub=5.0)
        m.addConstr(x + y == 5)
        m.setObjective(x, GRB.MAXIMIZE)
        m.optimize()
        assert x.X == pytest.approx(5.0, abs=TOL)
        m.setObjective(y, GRB.MAXIMIZE)
        m.optimize()
        assert y.X == pytest.approx(5.0, abs=TOL)

    def test_reoptimize_after_bound_change(self):
        m = Model()
        x = m.addVar(lb=0.0, ub=10.0)
        m.setObjective(x, GRB.MAXIMIZE)
        m.optimize()
        assert x.X == pytest.approx(10.0, abs=TOL)
        x.UB = 4.0
        m.optimize()
        assert x.X == pytest.approx(4.0, abs=TOL)


# ------------------------------------------------------------------ #
# Larger LP
# ------------------------------------------------------------------ #

class TestLPLarger:
    def test_5var_lp(self):
        """
        min  x0 + x1 + x2 + x3 + x4
        s.t. xi + x(i+1) >= 1  for i in 0..3
             xi >= 0
        """
        n = 5
        m = Model()
        from grbcompat import quicksum
        x = m.addVars(n, lb=0.0, name="x")
        m.setObjective(quicksum(x[i] for i in range(n)), GRB.MINIMIZE)
        for i in range(n - 1):
            m.addConstr(x[i] + x[i + 1] >= 1)
        m.optimize()
        assert m.Status == GRB.OPTIMAL
        assert m.ObjVal >= 0.0

    def test_transportation_lp(self):
        """
        2-supply, 3-demand transportation LP.
        supply = [30, 40], demand = [20, 25, 25]
        cost = [[2, 3, 1], [5, 4, 8]]
        min total cost.
        """
        supply = [30, 40]
        demand = [20, 25, 25]
        cost = [[2, 3, 1], [5, 4, 8]]
        m = Model("transport")
        from grbcompat import quicksum
        x = m.addVars(2, 3, lb=0.0, name="x")
        m.setObjective(
            quicksum(cost[i][j] * x[i, j] for i in range(2) for j in range(3)),
            GRB.MINIMIZE,
        )
        for i in range(2):
            m.addConstr(quicksum(x[i, j] for j in range(3)) <= supply[i])
        for j in range(3):
            m.addConstr(quicksum(x[i, j] for i in range(2)) >= demand[j])
        m.optimize()
        assert m.Status == GRB.OPTIMAL
        # Verify feasibility
        for j in range(3):
            assert sum(x[i, j].X for i in range(2)) >= demand[j] - TOL
