"""Constr class – a linear constraint backed by a HiGHS row."""

from __future__ import annotations


class Constr:
    """
    Represents a single linear constraint.

    Instances are returned by Model.addConstr() / Model.addConstrs() and
    should not be constructed directly.
    """

    __slots__ = ("_row_idx", "_model", "_name", "_sense", "_rhs")

    def __init__(self, row_idx: int, model, name: str = "", sense: str = "", rhs: float = 0.0):
        self._row_idx: int = row_idx
        self._model = model
        self._name: str = name
        self._sense: str = sense
        self._rhs: float = float(rhs)

    def _check_live(self) -> None:
        if self._row_idx < 0:
            raise RuntimeError(
                f"Constr '{self._name}' has been removed from the model."
            )

    # ------------------------------------------------------------------ #
    # Solution attributes
    # ------------------------------------------------------------------ #

    @property
    def Pi(self) -> float:
        """Dual value / shadow price (available after optimize())."""
        self._check_live()
        if self._model._solution is None:
            raise RuntimeError("No solution available – call model.optimize() first.")
        return float(self._model._solution.row_dual[self._row_idx])

    @property
    def Slack(self) -> float:
        """Constraint slack (available after optimize())."""
        self._check_live()
        if self._model._solution is None:
            raise RuntimeError("No solution available – call model.optimize() first.")
        activity = float(self._model._solution.row_value[self._row_idx])
        return self._rhs - activity

    # ------------------------------------------------------------------ #
    # Model attributes
    # ------------------------------------------------------------------ #

    @property
    def ConstrName(self) -> str:
        return self._name

    @ConstrName.setter
    def ConstrName(self, value: str) -> None:
        self._name = value

    @property
    def Sense(self) -> str:
        return self._sense

    @Sense.setter
    def Sense(self, value: str) -> None:
        self._check_live()
        import highspy
        _INF = highspy.kHighsInf
        self._sense = value
        if value == "<":
            self._model._h.changeRowBounds(self._row_idx, -_INF, self._rhs)
        elif value == ">":
            self._model._h.changeRowBounds(self._row_idx, self._rhs, _INF)
        else:  # "="
            self._model._h.changeRowBounds(self._row_idx, self._rhs, self._rhs)

    @property
    def RHS(self) -> float:
        return self._rhs

    @RHS.setter
    def RHS(self, value: float) -> None:
        self._check_live()
        import highspy
        self._rhs = float(value)
        # Update the row bounds in HiGHS
        if self._sense == "<":
            self._model._h.changeRowBounds(self._row_idx, -highspy.kHighsInf, self._rhs)
        elif self._sense == ">":
            self._model._h.changeRowBounds(self._row_idx, self._rhs, highspy.kHighsInf)
        else:
            self._model._h.changeRowBounds(self._row_idx, self._rhs, self._rhs)

    def __repr__(self) -> str:
        return f"<Constr '{self._name}' row={self._row_idx} sense='{self._sense}' rhs={self._rhs}>"
