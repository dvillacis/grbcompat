"""Linear expression and temporary constraint classes."""

from __future__ import annotations


class TempConstr:
    """Pending constraint created by comparison operators on Var / LinExpr."""

    __slots__ = ("lhs", "sense", "rhs")

    def __init__(self, lhs, sense: str, rhs):
        self.lhs = lhs
        self.sense = sense
        self.rhs = rhs


class LinExpr:
    """
    Linear expression: sum of (coeff * var) terms plus an optional constant.

    Arithmetic with Var and other LinExpr objects produces new LinExpr
    instances; comparison operators produce TempConstr objects ready to be
    passed to Model.addConstr().
    """

    def __init__(self, coeffs=None, vars=None, constant: float = 0.0):
        # _terms: {id(var): (var, coeff)}
        self._terms: dict = {}
        self.constant: float = float(constant)

        if vars is not None and coeffs is not None:
            for v, c in zip(vars, coeffs):
                self._add_term(v, c)
        elif coeffs is not None and vars is None:
            # Called as LinExpr(scalar) – treat as constant
            if isinstance(coeffs, (int, float)):
                self.constant = float(coeffs)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _add_term(self, var, coeff: float) -> None:
        coeff = float(coeff)
        if coeff == 0.0:
            return
        vid = id(var)
        if vid in self._terms:
            new_c = self._terms[vid][1] + coeff
            if new_c == 0.0:
                del self._terms[vid]
            else:
                self._terms[vid] = (var, new_c)
        else:
            self._terms[vid] = (var, coeff)

    def _copy(self) -> "LinExpr":
        result = LinExpr()
        result._terms = dict(self._terms)
        result.constant = self.constant
        return result

    # ------------------------------------------------------------------ #
    # gurobipy-compatible inspection methods
    # ------------------------------------------------------------------ #

    def size(self) -> int:
        return len(self._terms)

    def getConstant(self) -> float:
        return self.constant

    def getCoeff(self, i: int) -> float:
        return list(self._terms.values())[i][1]

    def getVar(self, i: int):
        return list(self._terms.values())[i][0]

    def getValue(self) -> float:
        """Evaluate expression using current solution values."""
        total = self.constant
        for _, (v, c) in self._terms.items():
            total += c * v.X
        return total

    def add(self, expr, mult: float = 1.0) -> None:
        """In-place addition (mutates self), mirroring gurobipy LinExpr.add."""
        if isinstance(expr, (int, float)):
            self.constant += mult * expr
        elif isinstance(expr, LinExpr):
            for vid, (v, c) in expr._terms.items():
                self._add_term(v, c * mult)
            self.constant += expr.constant * mult
        else:
            # Var
            self._add_term(expr, mult)

    def addTerms(self, coeffs, vars) -> None:
        """Add multiple terms in-place."""
        if isinstance(coeffs, (int, float)):
            coeffs = [coeffs]
            vars = [vars]
        for c, v in zip(coeffs, vars):
            self._add_term(v, c)

    # ------------------------------------------------------------------ #
    # Arithmetic operators
    # ------------------------------------------------------------------ #

    def __add__(self, other) -> "LinExpr":
        result = self._copy()
        if isinstance(other, (int, float)):
            result.constant += other
        elif isinstance(other, LinExpr):
            for _, (v, c) in other._terms.items():
                result._add_term(v, c)
            result.constant += other.constant
        elif hasattr(other, "_col_idx"):  # Var
            result._add_term(other, 1.0)
        else:
            return NotImplemented
        return result

    def __radd__(self, other) -> "LinExpr":
        if other == 0:
            return self._copy()
        return self.__add__(other)

    def __sub__(self, other) -> "LinExpr":
        result = self._copy()
        if isinstance(other, (int, float)):
            result.constant -= other
        elif isinstance(other, LinExpr):
            for _, (v, c) in other._terms.items():
                result._add_term(v, -c)
            result.constant -= other.constant
        elif hasattr(other, "_col_idx"):  # Var
            result._add_term(other, -1.0)
        else:
            return NotImplemented
        return result

    def __rsub__(self, other) -> "LinExpr":
        negated = self.__mul__(-1.0)
        if isinstance(other, (int, float)):
            negated.constant += other
        elif hasattr(other, "_col_idx"):  # Var
            negated._add_term(other, 1.0)
        elif isinstance(other, LinExpr):
            for _, (v, c) in other._terms.items():
                negated._add_term(v, c)
            negated.constant += other.constant
        else:
            return NotImplemented
        return negated

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            result = LinExpr()
            result._terms = {vid: (v, c * other) for vid, (v, c) in self._terms.items()}
            result.constant = self.constant * other
            return result
        if hasattr(other, "_col_idx"):  # Var — produce QuadExpr
            from grbcompat._quadexpr import QuadExpr
            q = QuadExpr()
            for _, (v, c) in self._terms.items():
                q._quad_terms.append((c, v, other))
            if self.constant != 0.0:
                q._lin_expr._add_term(other, self.constant)
            return q
        return NotImplemented

    def __rmul__(self, scalar) -> "LinExpr":
        return self.__mul__(scalar)

    def __truediv__(self, scalar) -> "LinExpr":
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        return self.__mul__(1.0 / scalar)

    def __neg__(self) -> "LinExpr":
        return self.__mul__(-1.0)

    # ------------------------------------------------------------------ #
    # Comparison operators → TempConstr
    # ------------------------------------------------------------------ #

    def __le__(self, rhs) -> TempConstr:
        from grbcompat._constants import GRB
        return TempConstr(self, GRB.LESS_EQUAL, rhs)

    def __ge__(self, rhs) -> TempConstr:
        from grbcompat._constants import GRB
        return TempConstr(self, GRB.GREATER_EQUAL, rhs)

    def __eq__(self, rhs) -> TempConstr:  # type: ignore[override]
        from grbcompat._constants import GRB
        return TempConstr(self, GRB.EQUAL, rhs)

    def __hash__(self):
        return id(self)

    def __repr__(self) -> str:
        parts = [f"{c}*{v.VarName}" for _, (v, c) in self._terms.items()]
        if self.constant:
            parts.append(str(self.constant))
        return " + ".join(parts) if parts else "0"
