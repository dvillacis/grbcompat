"""Var class – a decision variable backed by a HiGHS column."""

from __future__ import annotations

from grbcompat._constants import GRB
from grbcompat._expr import LinExpr, TempConstr


class Var:
    """
    Represents a single optimization variable.

    Instances are returned by Model.addVar() / Model.addVars() and should not
    be constructed directly. Arithmetic operators produce LinExpr objects;
    comparison operators produce TempConstr objects.
    """

    __slots__ = ("_col_idx", "_model", "_name")

    def __init__(self, col_idx: int, model, name: str = ""):
        self._col_idx: int = col_idx
        self._model = model
        self._name: str = name

    def _check_live(self) -> None:
        if self._col_idx < 0:
            raise RuntimeError(
                f"Var '{self._name}' has been removed from the model."
            )

    # ------------------------------------------------------------------ #
    # Solution attributes
    # ------------------------------------------------------------------ #

    @property
    def X(self) -> float:
        """Primal solution value (available after optimize())."""
        self._check_live()
        if self._model._solution is None:
            raise RuntimeError("No solution available – call model.optimize() first.")
        return float(self._model._solution.col_value[self._col_idx])

    @property
    def x(self) -> float:
        """Lowercase alias for X — accepts both .x and .X."""
        return self.X

    @property
    def RC(self) -> float:
        """Reduced cost (available after optimize())."""
        self._check_live()
        if self._model._solution is None:
            raise RuntimeError("No solution available – call model.optimize() first.")
        return float(self._model._solution.col_dual[self._col_idx])

    # ------------------------------------------------------------------ #
    # Model attributes (readable and writable)
    # ------------------------------------------------------------------ #

    @property
    def LB(self) -> float:
        self._check_live()
        return self._model._var_lbs[self._col_idx]

    @LB.setter
    def LB(self, value: float) -> None:
        self._check_live()
        import highspy
        lb = -highspy.kHighsInf if value <= -GRB.INFINITY / 2 else float(value)
        self._model._h.changeColBounds(self._col_idx, lb, self._model._var_ubs[self._col_idx])
        self._model._var_lbs[self._col_idx] = lb

    @property
    def UB(self) -> float:
        self._check_live()
        return self._model._var_ubs[self._col_idx]

    @UB.setter
    def UB(self, value: float) -> None:
        self._check_live()
        import highspy
        ub = highspy.kHighsInf if value >= GRB.INFINITY / 2 else float(value)
        self._model._h.changeColBounds(self._col_idx, self._model._var_lbs[self._col_idx], ub)
        self._model._var_ubs[self._col_idx] = ub

    @property
    def Obj(self) -> float:
        self._check_live()
        return self._model._var_objs[self._col_idx]

    @Obj.setter
    def Obj(self, value: float) -> None:
        self._check_live()
        self._model._h.changeColCost(self._col_idx, float(value))
        self._model._var_objs[self._col_idx] = float(value)

    @property
    def VarName(self) -> str:
        return self._name

    @VarName.setter
    def VarName(self, value: str) -> None:
        self._name = value

    @property
    def VType(self) -> str:
        self._check_live()
        return self._model._var_vtypes[self._col_idx]

    @property
    def Start(self) -> float:
        """MIP warm-start hint.  ``GRB.UNDEFINED`` means no hint is set."""
        self._check_live()
        return self._model._var_starts[self._col_idx]

    @Start.setter
    def Start(self, value: float) -> None:
        self._check_live()
        self._model._var_starts[self._col_idx] = float(value)

    @property
    def BranchPriority(self) -> int:
        """MIP branching priority.  Higher values are branched on first."""
        self._check_live()
        return self._model._var_branch_priorities[self._col_idx]

    @BranchPriority.setter
    def BranchPriority(self, value: int) -> None:
        self._check_live()
        self._model._var_branch_priorities[self._col_idx] = int(value)

    @VType.setter
    def VType(self, vtype: str) -> None:
        self._check_live()
        import highspy
        if vtype in (GRB.BINARY, GRB.INTEGER):
            self._model._h.changeColIntegrality(
                self._col_idx, highspy.HighsVarType.kInteger
            )
            if vtype == GRB.BINARY:
                self._model._h.changeColBounds(self._col_idx, 0.0, 1.0)
                self._model._var_lbs[self._col_idx] = 0.0
                self._model._var_ubs[self._col_idx] = 1.0
        else:
            self._model._h.changeColIntegrality(
                self._col_idx, highspy.HighsVarType.kContinuous
            )
        self._model._var_vtypes[self._col_idx] = vtype

    # ------------------------------------------------------------------ #
    # Arithmetic operators
    # ------------------------------------------------------------------ #

    def __add__(self, other) -> LinExpr:
        expr = LinExpr()
        expr._add_term(self, 1.0)
        if isinstance(other, (int, float)):
            expr.constant = float(other)
        elif isinstance(other, Var):
            expr._add_term(other, 1.0)
        elif isinstance(other, LinExpr):
            return other.__add__(self)
        else:
            return NotImplemented
        return expr

    def __radd__(self, other) -> LinExpr:
        if other == 0:
            expr = LinExpr()
            expr._add_term(self, 1.0)
            return expr
        return self.__add__(other)

    def __sub__(self, other) -> LinExpr:
        expr = LinExpr()
        expr._add_term(self, 1.0)
        if isinstance(other, (int, float)):
            expr.constant = -float(other)
        elif isinstance(other, Var):
            expr._add_term(other, -1.0)
        elif isinstance(other, LinExpr):
            for _, (v, c) in other._terms.items():
                expr._add_term(v, -c)
            expr.constant = -other.constant
        else:
            return NotImplemented
        return expr

    def __rsub__(self, other) -> LinExpr:
        expr = LinExpr()
        expr._add_term(self, -1.0)
        if isinstance(other, (int, float)):
            expr.constant = float(other)
        elif isinstance(other, Var):
            expr._add_term(other, 1.0)
        else:
            return NotImplemented
        return expr

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            expr = LinExpr()
            expr._add_term(self, float(other))
            return expr
        if hasattr(other, "_col_idx"):  # other is a Var
            from grbcompat._quadexpr import QuadExpr
            q = QuadExpr()
            q._quad_terms.append((1.0, self, other))
            return q
        return NotImplemented

    def __rmul__(self, scalar) -> LinExpr:
        return self.__mul__(scalar)

    def __truediv__(self, scalar) -> LinExpr:
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        return self.__mul__(1.0 / scalar)

    def __neg__(self) -> LinExpr:
        return self.__mul__(-1.0)

    # ------------------------------------------------------------------ #
    # Comparison operators → TempConstr
    # ------------------------------------------------------------------ #

    def __le__(self, rhs) -> TempConstr:
        expr = LinExpr()
        expr._add_term(self, 1.0)
        return TempConstr(expr, GRB.LESS_EQUAL, rhs)

    def __ge__(self, rhs) -> TempConstr:
        expr = LinExpr()
        expr._add_term(self, 1.0)
        return TempConstr(expr, GRB.GREATER_EQUAL, rhs)

    def __eq__(self, rhs) -> TempConstr:  # type: ignore[override]
        expr = LinExpr()
        expr._add_term(self, 1.0)
        return TempConstr(expr, GRB.EQUAL, rhs)

    def __hash__(self):
        return id(self)

    def __repr__(self) -> str:
        return f"<Var '{self._name}' col={self._col_idx}>"
