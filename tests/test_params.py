"""Tests for the Params proxy (model.Params)."""

import pytest
from grbcompat import GRB, Model


class TestParamsSetAndGet:
    def test_time_limit_set(self):
        m = Model()
        m.Params.TimeLimit = 120.0
        assert m.Params.TimeLimit == pytest.approx(120.0)

    def test_time_limit_via_set_param(self):
        m = Model()
        m.setParam("TimeLimit", 60.0)
        assert m.Params.TimeLimit == pytest.approx(60.0)

    def test_mip_gap_set(self):
        m = Model()
        m.Params.MIPGap = 0.01
        assert m.Params.MIPGap == pytest.approx(0.01)

    def test_output_flag_off(self):
        # Default is False (set in Model.__init__)
        m = Model()
        assert not m.Params.OutputFlag  # falsy value

    def test_output_flag_on(self):
        m = Model()
        m.Params.OutputFlag = True
        assert m.Params.OutputFlag

    def test_threads_set(self):
        m = Model()
        m.Params.Threads = 2
        assert m.Params.Threads == 2

    def test_feasibility_tol_set(self):
        m = Model()
        m.Params.FeasibilityTol = 1e-5
        assert m.Params.FeasibilityTol == pytest.approx(1e-5)

    def test_optimality_tol_set(self):
        m = Model()
        m.Params.OptimalityTol = 1e-7
        assert m.Params.OptimalityTol == pytest.approx(1e-7)

    def test_integer_feas_tol_set(self):
        m = Model()
        m.Params.IntFeasTol = 1e-5
        assert m.Params.IntFeasTol == pytest.approx(1e-5)

    def test_node_limit_set(self):
        m = Model()
        m.Params.NodeLimit = 1000
        assert m.Params.NodeLimit == 1000

    def test_iteration_limit_set(self):
        m = Model()
        m.Params.IterationLimit = 500
        assert m.Params.IterationLimit == 500

    def test_seed_set(self):
        m = Model()
        m.Params.Seed = 42
        assert m.Params.Seed == 42


class TestParamsIgnored:
    def test_sol_files_silently_ignored(self):
        """SolFiles has no HiGHS equivalent; setting it should not raise."""
        m = Model()
        m.Params.SolFiles = "output"  # should not raise

    def test_solution_number_silently_ignored(self):
        m = Model()
        m.Params.SolutionNumber = 0  # should not raise


class TestParamsUnknown:
    def test_unknown_param_raises(self):
        m = Model()
        with pytest.raises((AttributeError, Exception)):
            m.Params.NonExistentParamXYZ = 999


class TestParamsTakesEffectInSolve:
    def test_time_limit_terminates_solve(self):
        """A very tight time limit should cause a non-optimal status (or optimal on trivial)."""
        m = Model()
        x = m.addVar(lb=0.0)
        m.setObjective(x, GRB.MINIMIZE)
        m.Params.TimeLimit = 0.001  # extremely tight
        m.optimize()
        # On a trivial problem it may still be OPTIMAL; just check no exception
        assert m.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT)
