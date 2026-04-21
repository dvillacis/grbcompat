"""GenConstr — general constraint wrapper (big-M linearized)."""

from __future__ import annotations


class GenConstr:
    """
    Represents a general constraint attached to a Model.

    Supported types (all implemented via big-M linearization):
    - INDICATOR : if binvar == binval then lhs sense rhs
    - ABS       : resvar = |argvar|
    - MIN       : resvar = min(vars [, constant])
    - MAX       : resvar = max(vars [, constant])

    Attributes
    ----------
    _model          : back-reference to owning Model
    _type           : one of the class constants below
    _name           : user-supplied name (may be empty)
    _helper_vars    : auxiliary Var objects added during linearization
    _helper_constrs : auxiliary Constr objects added during linearization
    _removed        : True once this constraint has been removed from its model
    """

    __slots__ = (
        "_model",
        "_type",
        "_name",
        "_helper_vars",
        "_helper_constrs",
        "_removed",
    )

    INDICATOR = "INDICATOR"
    ABS = "ABS"
    MIN = "MIN"
    MAX = "MAX"

    def __init__(self, model, gctype: str, name: str = "") -> None:
        self._model = model
        self._type = gctype
        self._name = name
        self._helper_vars: list = []
        self._helper_constrs: list = []
        self._removed: bool = False

    # ------------------------------------------------------------------ #
    # Internal guard
    # ------------------------------------------------------------------ #

    def _check_live(self) -> None:
        if self._removed:
            raise RuntimeError(
                f"GenConstr '{self._name}' has been removed from the model."
            )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @property
    def index(self) -> int:
        """0-based position of this GenConstr in the model's gen-constr list."""
        self._check_live()
        return self._model._gen_constrs.index(self)

    def __repr__(self) -> str:
        return f"<GenConstr({self._type}) '{self._name}'>"
