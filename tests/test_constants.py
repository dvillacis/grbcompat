"""Tests for GRB constants."""

from grbcompat._constants import GRB


class TestGRBObjectiveSense:
    def test_minimize_is_1(self):
        assert GRB.MINIMIZE == 1

    def test_maximize_is_minus_1(self):
        assert GRB.MAXIMIZE == -1

    def test_senses_are_distinct(self):
        assert GRB.MINIMIZE != GRB.MAXIMIZE


class TestGRBInfinity:
    def test_infinity_is_large(self):
        assert GRB.INFINITY > 1e50

    def test_negative_infinity(self):
        assert -GRB.INFINITY < -1e50


class TestGRBVariableTypes:
    def test_continuous(self):
        assert GRB.CONTINUOUS == "C"

    def test_integer(self):
        assert GRB.INTEGER == "I"

    def test_binary(self):
        assert GRB.BINARY == "B"

    def test_semicont(self):
        assert GRB.SEMICONT == "S"

    def test_semiint(self):
        assert GRB.SEMIINT == "N"

    def test_all_distinct(self):
        vtypes = {GRB.CONTINUOUS, GRB.INTEGER, GRB.BINARY, GRB.SEMICONT, GRB.SEMIINT}
        assert len(vtypes) == 5


class TestGRBConstraintSenses:
    def test_less_equal(self):
        assert GRB.LESS_EQUAL == "<"

    def test_greater_equal(self):
        assert GRB.GREATER_EQUAL == ">"

    def test_equal(self):
        assert GRB.EQUAL == "="

    def test_all_distinct(self):
        senses = {GRB.LESS_EQUAL, GRB.GREATER_EQUAL, GRB.EQUAL}
        assert len(senses) == 3


class TestGRBStatusCodes:
    """Status codes are consistent between GRB and GRB.Status."""

    def test_loaded(self):
        assert GRB.LOADED == 1
        assert GRB.Status.LOADED == 1

    def test_optimal(self):
        assert GRB.OPTIMAL == 2
        assert GRB.Status.OPTIMAL == 2

    def test_infeasible(self):
        assert GRB.INFEASIBLE == 3
        assert GRB.Status.INFEASIBLE == 3

    def test_inf_or_unbd(self):
        assert GRB.INF_OR_UNBD == 4
        assert GRB.Status.INF_OR_UNBD == 4

    def test_unbounded(self):
        assert GRB.UNBOUNDED == 5
        assert GRB.Status.UNBOUNDED == 5

    def test_time_limit(self):
        assert GRB.TIME_LIMIT == 9
        assert GRB.Status.TIME_LIMIT == 9

    def test_all_top_level_match_status_class(self):
        attrs = [
            "LOADED", "OPTIMAL", "INFEASIBLE", "INF_OR_UNBD", "UNBOUNDED",
            "CUTOFF", "ITERATION_LIMIT", "NODE_LIMIT", "TIME_LIMIT",
            "SOLUTION_LIMIT", "INTERRUPTED", "NUMERIC", "SUBOPTIMAL",
            "INPROGRESS", "USER_OBJ_LIMIT",
        ]
        for attr in attrs:
            assert getattr(GRB, attr) == getattr(GRB.Status, attr), attr

    def test_all_status_codes_distinct(self):
        codes = [
            GRB.LOADED, GRB.OPTIMAL, GRB.INFEASIBLE, GRB.INF_OR_UNBD,
            GRB.UNBOUNDED, GRB.CUTOFF, GRB.ITERATION_LIMIT, GRB.NODE_LIMIT,
            GRB.TIME_LIMIT, GRB.SOLUTION_LIMIT, GRB.INTERRUPTED, GRB.NUMERIC,
            GRB.SUBOPTIMAL, GRB.INPROGRESS, GRB.USER_OBJ_LIMIT,
        ]
        assert len(set(codes)) == len(codes)


class TestGRBAttr:
    def test_obj_val(self):
        assert GRB.Attr.ObjVal == "ObjVal"

    def test_x(self):
        assert GRB.Attr.X == "X"

    def test_pi(self):
        assert GRB.Attr.Pi == "Pi"

    def test_slack(self):
        assert GRB.Attr.Slack == "Slack"

    def test_rc(self):
        assert GRB.Attr.RC == "RC"
