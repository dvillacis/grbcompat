"""QuadExpr and TempQConstr – quadratic expression classes."""

from __future__ import annotations


class TempQConstr:
    """Pending quadratic constraint created by comparison operators on QuadExpr."""

    __slots__ = ("lhs", "sense", "rhs")

    def __init__(self, lhs, sense: str, rhs):
        self.lhs = lhs
        self.sense = sense
        self.rhs = rhs


class QuadExpr:
    """
    Quadratic expression: a sum of (coeff * var1 * var2) terms plus a linear part.

    Instances are produced by multiplying two Var objects together or by arithmetic
    on existing QuadExpr objects.  Comparison operators produce TempQConstr objects
    ready to be passed to Model.addQConstr() (which raises NotImplementedError —
    HiGHS supports quadratic objectives only).

    The quadratic terms are stored as a list of (coeff, var1, var2) triples.
    Duplicate entries for the same pair are merged at Hessian-build time inside
    Model.setObjective(), not here.  This keeps arithmetic O(n) and avoids the
    overhead of a keyed dict for the typical small-to-medium expression sizes.
    """

    def __init__(self):
        from grbcompat._expr import LinExpr
        self._quad_terms: list = []   # [(coeff, var1, var2), ...]
        self._lin_expr: LinExpr = LinExpr()

    # ------------------------------------------------------------------ #
    # gurobipy-compatible inspection API
    # ------------------------------------------------------------------ #

    def size(self) -> int:
        """Number of quadratic terms."""
        return len(self._quad_terms)

    def getCoeff(self, i: int) -> float:
        return self._quad_terms[i][0]

    def getVar1(self, i: int):
        return self._quad_terms[i][1]

    def getVar2(self, i: int):
        return self._quad_terms[i][2]

    def getLinExpr(self):
        """Return a copy of the linear part."""
        return self._lin_expr._copy()

    def getValue(self) -> float:
        """Evaluate the expression using current solution values."""
        total = self._lin_expr.getValue()
        for coeff, v1, v2 in self._quad_terms:
            total += coeff * v1.X * v2.X
        return total

    # ------------------------------------------------------------------ #
    # Internal copy helper
    # ------------------------------------------------------------------ #

    def _copy(self) -> "QuadExpr":
        result = QuadExpr()
        result._quad_terms = list(self._quad_terms)
        result._lin_expr = self._lin_expr._copy()
        return result

    # ------------------------------------------------------------------ #
    # Arithmetic operators
    # ------------------------------------------------------------------ #

    def __add__(self, other) -> "QuadExpr":
        result = self._copy()
        if isinstance(other, QuadExpr):
            result._quad_terms.extend(other._quad_terms)
            result._lin_expr = result._lin_expr + other._lin_expr
        elif isinstance(other, (int, float)):
            result._lin_expr.constant += other
        elif hasattr(other, "_terms"):  # LinExpr
            result._lin_expr = result._lin_expr + other
        elif hasattr(other, "_col_idx"):  # Var
            result._lin_expr._add_term(other, 1.0)
        else:
            return NotImplemented
        return result

    def __radd__(self, other) -> "QuadExpr":
        if other == 0:
            return self._copy()
        return self.__add__(other)

    def __sub__(self, other) -> "QuadExpr":
        result = self._copy()
        if isinstance(other, QuadExpr):
            result._quad_terms.extend((-c, v1, v2) for c, v1, v2 in other._quad_terms)
            result._lin_expr = result._lin_expr - other._lin_expr
        elif isinstance(other, (int, float)):
            result._lin_expr.constant -= other
        elif hasattr(other, "_terms"):  # LinExpr
            result._lin_expr = result._lin_expr - other
        elif hasattr(other, "_col_idx"):  # Var
            result._lin_expr._add_term(other, -1.0)
        else:
            return NotImplemented
        return result

    def __rsub__(self, other) -> "QuadExpr":
        negated = self.__neg__()
        if isinstance(other, (int, float)):
            negated._lin_expr.constant += other
        elif hasattr(other, "_terms"):  # LinExpr
            negated._lin_expr = other + negated._lin_expr
        elif hasattr(other, "_col_idx"):  # Var
            negated._lin_expr._add_term(other, 1.0)
        else:
            return NotImplemented
        return negated

    def __mul__(self, scalar) -> "QuadExpr":
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        result = QuadExpr()
        result._quad_terms = [(c * scalar, v1, v2) for c, v1, v2 in self._quad_terms]
        result._lin_expr = self._lin_expr * scalar
        return result

    def __rmul__(self, scalar) -> "QuadExpr":
        return self.__mul__(scalar)

    def __truediv__(self, scalar) -> "QuadExpr":
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        return self.__mul__(1.0 / scalar)

    def __neg__(self) -> "QuadExpr":
        return self.__mul__(-1.0)

    # ------------------------------------------------------------------ #
    # Comparison operators → TempQConstr
    # ------------------------------------------------------------------ #

    def __le__(self, rhs) -> TempQConstr:
        from grbcompat._constants import GRB
        return TempQConstr(self, GRB.LESS_EQUAL, rhs)

    def __ge__(self, rhs) -> TempQConstr:
        from grbcompat._constants import GRB
        return TempQConstr(self, GRB.GREATER_EQUAL, rhs)

    def __eq__(self, rhs) -> TempQConstr:  # type: ignore[override]
        from grbcompat._constants import GRB
        return TempQConstr(self, GRB.EQUAL, rhs)

    def __hash__(self):
        return id(self)

    def __repr__(self) -> str:
        parts = [f"{c}*{v1.VarName}*{v2.VarName}" for c, v1, v2 in self._quad_terms]
        lin_repr = repr(self._lin_expr)
        if lin_repr != "0":
            parts.append(lin_repr)
        return " + ".join(parts) if parts else "0"
