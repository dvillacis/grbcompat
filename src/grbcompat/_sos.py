"""SOS — Special Ordered Set constraint wrapper."""

from __future__ import annotations


class SOS:
    """
    Represents a Special Ordered Set constraint attached to a Model.

    Only SOS1 is supported (via big-M linearization).  SOS2 raises
    NotImplementedError at the addSOS() call site.

    Attributes
    ----------
    _model      : back-reference to owning Model
    _sostype    : 1 (SOS_TYPE1)
    _vars       : member variables (original set, not helpers)
    _weights    : ordering weights for the member variables
    _name       : user-supplied name (may be empty)
    _helper_vars    : binary indicator Var objects added during linearization
    _helper_constrs : big-M and sum Constr objects added during linearization
    _removed    : True once the SOS has been removed from its model
    """

    __slots__ = (
        "_model",
        "_sostype",
        "_vars",
        "_weights",
        "_name",
        "_helper_vars",
        "_helper_constrs",
        "_removed",
    )

    def __init__(self, model, sostype: int, vars: list, weights: list, name: str = "") -> None:
        self._model = model
        self._sostype = sostype
        self._vars = list(vars)
        self._weights = list(weights)
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
                f"SOS '{self._name}' has been removed from the model."
            )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @property
    def index(self) -> int:
        """0-based position of this SOS in the model's SOS list."""
        self._check_live()
        return self._model._sos_list.index(self)

    def __repr__(self) -> str:
        t = "SOS1" if self._sostype == 1 else "SOS2"
        return f"<{t} '{self._name}' nvars={len(self._vars)}>"
