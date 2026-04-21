"""tupledict – a dict subclass mirroring gurobipy.tupledict."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from grbcompat._expr import LinExpr


class tupledict(dict):
    """
    Dictionary subclass with index-filtering and expression-building helpers,
    mirroring the gurobipy.tupledict interface.

    Keys may be scalars or tuples. ``select`` and ``sum`` accept wildcard ``'*'``
    patterns to filter entries.
    """

    def select(self, *pattern) -> "tupledict":
        """
        Return a tupledict of entries whose keys match *pattern*.

        Use ``'*'`` as a per-position wildcard.
        """
        result: tupledict = tupledict()
        for key, val in self.items():
            key_tuple = key if isinstance(key, tuple) else (key,)
            if len(pattern) != len(key_tuple):
                continue
            if all(p == "*" or p == k for p, k in zip(pattern, key_tuple)):
                result[key] = val
        return result

    def sum(self, *pattern) -> "LinExpr":
        """Sum all matching values into a LinExpr."""
        from grbcompat._expr import LinExpr

        selected = self.select(*pattern) if pattern else self
        result = LinExpr()
        for val in selected.values():
            result = result + val
        return result

    def prod(self, coeff_dict: dict, *pattern) -> "LinExpr":
        """Weighted sum of matching values: sum(coeff[key] * var[key])."""
        from grbcompat._expr import LinExpr

        selected = self.select(*pattern) if pattern else self
        result = LinExpr()
        for key, val in selected.items():
            if key in coeff_dict:
                result = result + coeff_dict[key] * val
        return result

    def __repr__(self) -> str:
        return f"tupledict({dict.__repr__(self)})"
