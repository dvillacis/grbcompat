"""Model class – the central object mirroring gurobipy.Model."""

from __future__ import annotations

import bisect
import itertools

try:
    import highspy
except ImportError as exc:
    raise ImportError(
        "highspy is required by grbcompat. "
        "Install it with:  pip install highspy"
    ) from exc

import math

from grbcompat._constants import GRB
from grbcompat._constr import Constr
from grbcompat._expr import LinExpr, TempConstr
from grbcompat._genconstr import GenConstr
from grbcompat._params import Params
from grbcompat._quadexpr import QuadExpr
from grbcompat._sos import SOS
from grbcompat._tupledict import tupledict
from grbcompat._var import Var

_DEFAULT_BIGM = 1e6

_INF = highspy.kHighsInf


# ------------------------------------------------------------------ #
# Internal helpers
# ------------------------------------------------------------------ #

def _to_linexpr(expr) -> LinExpr:
    """Coerce Var, LinExpr, or numeric constant to LinExpr."""
    if isinstance(expr, LinExpr):
        return expr._copy()
    if isinstance(expr, Var):
        result = LinExpr()
        result._add_term(expr, 1.0)
        return result
    if isinstance(expr, (int, float)):
        result = LinExpr()
        result.constant = float(expr)
        return result
    raise TypeError(f"Cannot convert {type(expr).__name__!r} to LinExpr")


def _map_highs_status(status) -> int:
    ms = highspy.HighsModelStatus
    _map = {
        ms.kOptimal: GRB.OPTIMAL,
        ms.kInfeasible: GRB.INFEASIBLE,
        ms.kUnbounded: GRB.UNBOUNDED,
        ms.kUnboundedOrInfeasible: GRB.INF_OR_UNBD,
        ms.kObjectiveBound: GRB.USER_OBJ_LIMIT,
        ms.kObjectiveTarget: GRB.USER_OBJ_LIMIT,
        ms.kSolutionLimit: GRB.SOLUTION_LIMIT,
        ms.kIterationLimit: GRB.ITERATION_LIMIT,
        ms.kTimeLimit: GRB.TIME_LIMIT,
        ms.kUnknown: GRB.INPROGRESS,
        ms.kNotset: GRB.LOADED,
        ms.kModelEmpty: GRB.LOADED,
        ms.kLoadError: GRB.LOADED,
        ms.kModelError: GRB.NUMERIC,
        ms.kSolveError: GRB.NUMERIC,
    }
    return _map.get(status, GRB.LOADED)


# ------------------------------------------------------------------ #
# Model
# ------------------------------------------------------------------ #

class Model:
    """
    Drop-in replacement for ``gurobipy.Model``, backed by the HiGHS solver.

    The public API mirrors gurobipy as closely as possible so that existing
    code only needs::

        import grbcompat   # patches sys.modules['gurobipy']

    added at the top of the file.
    """

    def __init__(self, name: str = "", env=None):
        self._h = highspy.Highs()
        self._h.setOptionValue("output_flag", False)
        self._name: str = name

        # Per-variable metadata (indexed by col_idx)
        self._vars: list[Var] = []
        self._var_lbs: list[float] = []
        self._var_ubs: list[float] = []
        self._var_objs: list[float] = []
        self._var_vtypes: list[str] = []
        self._var_starts: list[float] = []          # GRB.UNDEFINED = no hint
        self._var_branch_priorities: list[int] = []

        # Per-constraint metadata (indexed by row_idx)
        self._constrs: list[Constr] = []

        # SOS constraints (big-M linearized)
        self._sos_list: list[SOS] = []

        # General constraints (big-M linearized)
        self._gen_constrs: list[GenConstr] = []

        self._status: int = GRB.LOADED
        self._solution = None
        self._obj_sense: int = GRB.MINIMIZE
        self._obj_offset: float = 0.0
        self._has_quad: bool = False    # True when a quadratic objective is active

        # Callback state — only meaningful during an optimize(callback) run
        self._cb_where: int = -1
        self._cb_data_out = None    # HiGHS HighsCallbackDataOut (live during callback)
        self._cb_data_in = None     # HiGHS HighsCallbackDataIn  (live during callback)
        self._cb_sol_count: int = 0
        self._cb_lazy_queue: list = []

        self.Params: Params = Params(self._h)

    # ------------------------------------------------------------------ #
    # Variable management
    # ------------------------------------------------------------------ #

    def addVar(
        self,
        lb: float = 0.0,
        ub: float = GRB.INFINITY,
        obj: float = 0.0,
        vtype: str = GRB.CONTINUOUS,
        name: str = "",
        column=None,
    ) -> Var:
        _lb = -_INF if lb <= -GRB.INFINITY / 2 else float(lb)
        _ub = _INF if ub >= GRB.INFINITY / 2 else float(ub)

        if vtype == GRB.BINARY:
            _lb, _ub = 0.0, 1.0

        self._h.addVar(_lb, _ub)
        col_idx = self._h.getNumCol() - 1

        if obj != 0.0:
            self._h.changeColCost(col_idx, float(obj))

        if vtype in (GRB.BINARY, GRB.INTEGER):
            self._h.changeColIntegrality(col_idx, highspy.HighsVarType.kInteger)

        self._var_lbs.append(_lb)
        self._var_ubs.append(_ub)
        self._var_objs.append(float(obj))
        self._var_vtypes.append(vtype)
        self._var_starts.append(GRB.UNDEFINED)
        self._var_branch_priorities.append(0)

        var = Var(col_idx, self, name=name)
        self._vars.append(var)
        return var

    def addVars(
        self,
        *args,
        lb: float | dict = 0.0,
        ub: float | dict = GRB.INFINITY,
        obj: float | dict = 0.0,
        vtype: str = GRB.CONTINUOUS,
        name: str = "",
    ) -> tupledict:
        if not args:
            raise ValueError("addVars() requires at least one index argument")

        index_sets: list[list] = []
        for a in args:
            if isinstance(a, int):
                index_sets.append(list(range(a)))
            elif hasattr(a, "__iter__"):
                index_sets.append(list(a))
            else:
                raise TypeError(f"Invalid index argument type: {type(a).__name__!r}")

        keys = index_sets[0] if len(index_sets) == 1 else list(itertools.product(*index_sets))

        result: tupledict = tupledict()
        for key in keys:
            if isinstance(key, tuple):
                key_str = ",".join(str(k) for k in key)
            else:
                key_str = str(key)
            var_name = f"{name}[{key_str}]" if name else ""

            _lb = lb[key] if isinstance(lb, dict) else lb
            _ub = ub[key] if isinstance(ub, dict) else ub
            _obj = obj[key] if isinstance(obj, dict) else obj

            var = self.addVar(lb=_lb, ub=_ub, obj=_obj, vtype=vtype, name=var_name)
            result[key] = var

        return result

    # ------------------------------------------------------------------ #
    # Constraint management
    # ------------------------------------------------------------------ #

    def addConstr(self, lhs_or_tc, sense=None, rhs=None, name: str = "") -> Constr:
        if isinstance(lhs_or_tc, TempConstr):
            tc = lhs_or_tc
            lhs_expr = _to_linexpr(tc.lhs)
            _sense = tc.sense
            _rhs = tc.rhs
        else:
            lhs_expr = _to_linexpr(lhs_or_tc)
            _sense = sense
            _rhs = rhs if rhs is not None else 0.0

        # If rhs contains variables, move them to lhs
        if isinstance(_rhs, (Var, LinExpr)):
            rhs_expr = _to_linexpr(_rhs)
            lhs_expr = lhs_expr - rhs_expr
            _rhs_val = 0.0
        else:
            _rhs_val = float(_rhs) if _rhs is not None else 0.0

        # Move constant from lhs to rhs
        _rhs_val -= lhs_expr.constant

        cols = [var._col_idx for _, (var, _) in lhs_expr._terms.items()]
        coeffs = [float(c) for _, (_, c) in lhs_expr._terms.items()]

        if _sense == GRB.LESS_EQUAL:
            row_lb, row_ub = -_INF, _rhs_val
        elif _sense == GRB.GREATER_EQUAL:
            row_lb, row_ub = _rhs_val, _INF
        elif _sense == GRB.EQUAL:
            row_lb, row_ub = _rhs_val, _rhs_val
        else:
            raise ValueError(f"Unknown constraint sense: {_sense!r}")

        self._h.addRow(row_lb, row_ub, len(cols), cols, coeffs)
        row_idx = self._h.getNumRow() - 1

        constr = Constr(row_idx, self, name=name, sense=_sense, rhs=_rhs_val)
        self._constrs.append(constr)
        return constr

    def addLConstr(self, lhs, sense=None, rhs=None, name: str = "") -> Constr:
        """Alias for addConstr (linear constraints only)."""
        return self.addConstr(lhs, sense=sense, rhs=rhs, name=name)

    def addConstrs(self, generator, name: str = "") -> tupledict:
        result: tupledict = tupledict()
        for i, tc in enumerate(generator):
            constr_name = f"{name}[{i}]" if name else ""
            result[i] = self.addConstr(tc, name=constr_name)
        return result

    def addSOS(
        self,
        sostype: int,
        vars: list,
        weights: list | None = None,
        name: str = "",
    ) -> SOS:
        """
        Add a Special Ordered Set constraint.

        Only SOS1 is supported.  SOS2 raises NotImplementedError.

        Parameters
        ----------
        sostype : int
            GRB.SOS_TYPE1 (= 1).  GRB.SOS_TYPE2 raises NotImplementedError.
        vars : list[Var]
            Member variables.  Each must have a finite upper bound (required
            for the big-M linearization).
        weights : list[float] | None
            Ordering weights.  Defaults to [1, 2, ..., n].
        name : str
            Optional name for the SOS constraint.

        Returns
        -------
        SOS
        """
        if sostype == GRB.SOS_TYPE2:
            raise NotImplementedError(
                "SOS2 constraints are not supported: HiGHS has no native SOS2 "
                "API and the linearized reformulation is not yet implemented."
            )
        if sostype != GRB.SOS_TYPE1:
            raise ValueError(
                f"Unknown SOS type {sostype!r}. Use GRB.SOS_TYPE1 or GRB.SOS_TYPE2."
            )

        n = len(vars)
        if weights is None:
            weights = list(range(1, n + 1))
        if len(weights) != n:
            raise ValueError("len(weights) must equal len(vars)")

        # Validate finite upper bounds (required for big-M)
        ubs = []
        for v in vars:
            ub = v.UB
            if ub >= GRB.INFINITY / 2:
                raise ValueError(
                    f"Variable '{v.VarName}' has no finite upper bound. "
                    "SOS1 linearization requires finite upper bounds on all "
                    "member variables."
                )
            ubs.append(ub)

        sos = SOS(self, sostype, vars, weights, name=name)
        prefix = f"_sos1_{name}" if name else f"_sos1_{len(self._sos_list)}"

        # Binary indicator b_i for each member variable, plus big-M constraint
        for i, (v, ub) in enumerate(zip(vars, ubs)):
            bv = self.addVar(vtype=GRB.BINARY, name=f"{prefix}_b{i}")
            sos._helper_vars.append(bv)
            c = self.addConstr(v <= ub * bv, name=f"{prefix}_bm{i}")
            sos._helper_constrs.append(c)

        # sum(b_i) <= 1 — at most one indicator active
        sum_expr = LinExpr()
        for bv in sos._helper_vars:
            sum_expr = sum_expr + bv
        sos._helper_constrs.append(
            self.addConstr(sum_expr <= 1, name=f"{prefix}_sum")
        )

        self._sos_list.append(sos)
        return sos

    @property
    def NumSOS(self) -> int:
        """Number of SOS constraints in the model."""
        return len(self._sos_list)

    def getSOSs(self) -> list:
        """Return a list of all SOS constraint objects."""
        return list(self._sos_list)

    # ------------------------------------------------------------------ #
    # General constraints
    # ------------------------------------------------------------------ #

    def _bigm_for_expr(self, lhs_expr: LinExpr, rhs_val: float) -> float:
        """
        Estimate a big-M for the constraint ``lhs_expr <= rhs_val``.

        Computes the maximum over-estimate of ``max(lhs_vars) - eff_rhs`` and
        ``eff_rhs - min(lhs_vars)`` from variable bounds.  Falls back to
        ``_DEFAULT_BIGM`` if any bound is infinite.
        """
        eff_rhs = rhs_val - lhs_expr.constant
        try:
            max_lhs = sum(
                c * (v.UB if c > 0 else v.LB)
                for _, (v, c) in lhs_expr._terms.items()
            )
            min_lhs = sum(
                c * (v.LB if c > 0 else v.UB)
                for _, (v, c) in lhs_expr._terms.items()
            )
            if math.isinf(max_lhs) or math.isinf(min_lhs):
                return _DEFAULT_BIGM
            return max(abs(max_lhs - eff_rhs), abs(min_lhs - eff_rhs), 1.0) * 1.1
        except Exception:
            return _DEFAULT_BIGM

    def _bigm_for_var(self, var) -> float:
        """Return a big-M derived from ``var``'s bounds."""
        lb, ub = var.LB, var.UB
        if math.isinf(lb) or math.isinf(ub):
            return _DEFAULT_BIGM
        return max(abs(lb), abs(ub), 1.0) * 2.0

    def addGenConstrIndicator(
        self,
        binvar,
        binval: int,
        lhs_or_tc,
        sense=None,
        rhs=None,
        name: str = "",
    ) -> GenConstr:
        """
        Add an indicator constraint: ``if binvar == binval, then lhs sense rhs``.

        Linearized via big-M: the enforced linear constraint is relaxed by
        ``M * (1 - activation_term)`` when the indicator is inactive.

        Parameters
        ----------
        binvar  : binary Var (the indicator variable)
        binval  : 0 or 1
        lhs_or_tc : LinExpr, Var, or TempConstr (e.g. ``x + y <= 5``)
        sense   : GRB.LESS_EQUAL / GREATER_EQUAL / EQUAL (if not using TempConstr)
        rhs     : float (if not using TempConstr)
        name    : optional name
        """
        if isinstance(lhs_or_tc, TempConstr):
            tc = lhs_or_tc
            lhs_expr = _to_linexpr(tc.lhs)
            _sense = tc.sense
            _rhs_raw = tc.rhs
        else:
            lhs_expr = _to_linexpr(lhs_or_tc)
            _sense = sense
            _rhs_raw = rhs if rhs is not None else 0.0

        if isinstance(_rhs_raw, (Var, LinExpr)):
            raise ValueError(
                "addGenConstrIndicator: the RHS of the enforced constraint "
                "must be a numeric constant, not a variable expression."
            )
        _rhs = float(_rhs_raw)

        M = self._bigm_for_expr(lhs_expr, _rhs)
        gc = GenConstr(self, GenConstr.INDICATOR, name=name)
        prefix = name or f"_ind_{len(self._gen_constrs)}"

        # For sense LE (b=binval → lhs <= rhs):
        #   binval=1: lhs + M*b <= rhs + M  (equiv: lhs <= rhs + M*(1-b))
        #   binval=0: lhs - M*b <= rhs      (equiv: lhs <= rhs + M*b)
        if _sense in (GRB.LESS_EQUAL, GRB.EQUAL):
            if binval == 1:
                c = self.addConstr(
                    lhs_expr + M * binvar <= _rhs + M, name=f"{prefix}_le"
                )
            else:
                c = self.addConstr(
                    lhs_expr - M * binvar <= _rhs, name=f"{prefix}_le"
                )
            gc._helper_constrs.append(c)

        # For sense GE (b=binval → lhs >= rhs):
        #   binval=1: lhs - M*b >= rhs - M  (equiv: lhs >= rhs - M*(1-b))
        #   binval=0: lhs + M*b >= rhs      (equiv: lhs >= rhs - M*b)
        if _sense in (GRB.GREATER_EQUAL, GRB.EQUAL):
            if binval == 1:
                c = self.addConstr(
                    lhs_expr - M * binvar >= _rhs - M, name=f"{prefix}_ge"
                )
            else:
                c = self.addConstr(
                    lhs_expr + M * binvar >= _rhs, name=f"{prefix}_ge"
                )
            gc._helper_constrs.append(c)

        self._gen_constrs.append(gc)
        return gc

    def addGenConstrAbs(
        self,
        resvar,
        argvar,
        name: str = "",
    ) -> GenConstr:
        """
        Add an absolute-value constraint: ``resvar = |argvar|``.

        Introduces one binary variable ``z`` (1 when argvar >= 0) and four
        linear constraints that together enforce the equality.  Big-M is
        derived from ``argvar``'s bounds where possible.
        """
        M = self._bigm_for_var(argvar)
        gc = GenConstr(self, GenConstr.ABS, name=name)
        prefix = name or f"_abs_{len(self._gen_constrs)}"

        z = self.addVar(vtype=GRB.BINARY, name=f"{prefix}_z")
        gc._helper_vars.append(z)

        # resvar >= argvar  (always)
        gc._helper_constrs.append(
            self.addConstr(resvar - argvar >= 0, name=f"{prefix}_lb1")
        )
        # resvar >= -argvar  (always)
        gc._helper_constrs.append(
            self.addConstr(resvar + argvar >= 0, name=f"{prefix}_lb2")
        )
        # z=1 → resvar <= argvar:  resvar - argvar + M*z <= M
        gc._helper_constrs.append(
            self.addConstr(resvar - argvar + M * z <= M, name=f"{prefix}_ub1")
        )
        # z=0 → resvar <= -argvar: resvar + argvar - M*z <= 0
        gc._helper_constrs.append(
            self.addConstr(resvar + argvar - M * z <= 0, name=f"{prefix}_ub2")
        )

        self._gen_constrs.append(gc)
        return gc

    def addGenConstrMin(
        self,
        resvar,
        argvars: list,
        constant: float = GRB.INFINITY,
        name: str = "",
    ) -> GenConstr:
        """
        Add a minimum constraint: ``resvar = min(argvars [, constant])``.

        Requires each member variable to have a finite upper bound (used for
        big-M).  Introduces one binary per candidate value.

        Parameters
        ----------
        resvar   : result variable
        argvars  : list of Var
        constant : optional numeric value included in the minimum
        name     : optional name
        """
        gc = GenConstr(self, GenConstr.MIN, name=name)
        prefix = name or f"_min_{len(self._gen_constrs)}"

        # Candidates: (var_or_None, value_or_None, label)
        candidates: list = [(v, None, f"v{i}") for i, v in enumerate(argvars)]
        const_active = constant < GRB.INFINITY / 2
        if const_active:
            candidates.append((None, float(constant), "c"))

        # Compute a global M across all candidates
        M_vals = []
        for var, val, _ in candidates:
            if var is not None:
                m = self._bigm_for_var(var)
            else:
                m = abs(val) * 2 + 1 if val is not None else _DEFAULT_BIGM
            M_vals.append(m)
        M = max(M_vals) if M_vals else _DEFAULT_BIGM

        # resvar <= v_i for all candidates (and <= constant if active)
        for var, val, lbl in candidates:
            if var is not None:
                gc._helper_constrs.append(
                    self.addConstr(resvar <= var, name=f"{prefix}_ub_{lbl}")
                )
            else:
                gc._helper_constrs.append(
                    self.addConstr(resvar <= val, name=f"{prefix}_ub_{lbl}")
                )

        # Binary z_i: exactly one equals 1, meaning that candidate is the minimum
        z_vars = []
        for _, _, lbl in candidates:
            z = self.addVar(vtype=GRB.BINARY, name=f"{prefix}_z_{lbl}")
            gc._helper_vars.append(z)
            z_vars.append(z)

        # sum(z_i) == 1
        gc._helper_constrs.append(
            self.addConstr(
                sum(z_vars, LinExpr()) == 1, name=f"{prefix}_sum"
            )
        )

        # z_i=1 → resvar >= candidate_i:  resvar >= v_i - M*(1-z_i)
        # rearranged: resvar - v_i - M*z_i >= -M
        # z=1: resvar - v_i - M >= -M  →  resvar >= v_i  ✓
        # z=0: resvar - v_i    >= -M   →  resvar >= v_i - M  (relaxed)  ✓
        for (var, val, lbl), z in zip(candidates, z_vars):
            if var is not None:
                gc._helper_constrs.append(
                    self.addConstr(
                        resvar - var - M * z >= -M, name=f"{prefix}_lb_{lbl}"
                    )
                )
            else:
                # resvar - M*z >= val - M
                gc._helper_constrs.append(
                    self.addConstr(
                        resvar - M * z >= float(val) - M, name=f"{prefix}_lb_{lbl}"
                    )
                )

        self._gen_constrs.append(gc)
        return gc

    def addGenConstrMax(
        self,
        resvar,
        argvars: list,
        constant: float = -GRB.INFINITY,
        name: str = "",
    ) -> GenConstr:
        """
        Add a maximum constraint: ``resvar = max(argvars [, constant])``.

        Symmetric to ``addGenConstrMin``: resvar is at least all candidates, and
        the binary selection forces equality to the largest one.
        """
        gc = GenConstr(self, GenConstr.MAX, name=name)
        prefix = name or f"_max_{len(self._gen_constrs)}"

        candidates: list = [(v, None, f"v{i}") for i, v in enumerate(argvars)]
        const_active = constant > -GRB.INFINITY / 2
        if const_active:
            candidates.append((None, float(constant), "c"))

        M_vals = []
        for var, val, _ in candidates:
            if var is not None:
                m = self._bigm_for_var(var)
            else:
                m = abs(val) * 2 + 1 if val is not None else _DEFAULT_BIGM
            M_vals.append(m)
        M = max(M_vals) if M_vals else _DEFAULT_BIGM

        # resvar >= v_i for all candidates
        for var, val, lbl in candidates:
            if var is not None:
                gc._helper_constrs.append(
                    self.addConstr(resvar >= var, name=f"{prefix}_lb_{lbl}")
                )
            else:
                gc._helper_constrs.append(
                    self.addConstr(resvar >= val, name=f"{prefix}_lb_{lbl}")
                )

        # Binary z_i: exactly one equals 1, marking the maximum
        z_vars = []
        for _, _, lbl in candidates:
            z = self.addVar(vtype=GRB.BINARY, name=f"{prefix}_z_{lbl}")
            gc._helper_vars.append(z)
            z_vars.append(z)

        # sum(z_i) == 1
        gc._helper_constrs.append(
            self.addConstr(
                sum(z_vars, LinExpr()) == 1, name=f"{prefix}_sum"
            )
        )

        # z_i=1 → resvar <= candidate_i:  resvar <= v_i + M*(1-z_i)
        # rearranged: resvar - v_i + M*z_i <= M
        # z=1: resvar - v_i + M <= M  →  resvar <= v_i  ✓
        # z=0: resvar - v_i    <= M   →  resvar <= v_i + M  (relaxed)  ✓
        for (var, val, lbl), z in zip(candidates, z_vars):
            if var is not None:
                gc._helper_constrs.append(
                    self.addConstr(
                        resvar - var + M * z <= M, name=f"{prefix}_ub_{lbl}"
                    )
                )
            else:
                # resvar + M*z <= val + M
                gc._helper_constrs.append(
                    self.addConstr(
                        resvar + M * z <= float(val) + M, name=f"{prefix}_ub_{lbl}"
                    )
                )

        self._gen_constrs.append(gc)
        return gc

    @property
    def NumGenConstrs(self) -> int:
        """Number of general constraints in the model."""
        return len(self._gen_constrs)

    def getGenConstrs(self) -> list:
        """Return a list of all GenConstr objects."""
        return list(self._gen_constrs)

    def addQConstr(self, constr, name: str = "") -> None:
        """
        Not supported: HiGHS supports quadratic objectives but not quadratic
        constraints.  Calls to this method always raise NotImplementedError.
        """
        raise NotImplementedError(
            "addQConstr() is not supported: HiGHS only handles quadratic objectives, "
            "not quadratic constraints."
        )

    # ------------------------------------------------------------------ #
    # Objective
    # ------------------------------------------------------------------ #

    def _build_hessian(self, quad_terms):
        """
        Convert QuadExpr quadratic terms to upper-triangular CSC Hessian arrays.

        HiGHS represents the QP objective as ``0.5 * x' * Q * x``, so:
        - Diagonal term (var == var): Q[i,i] = 2 * coeff
        - Off-diagonal term (var1 != var2): Q[min_idx, max_idx] = coeff

        Returns (start, index, value) lists for use with passHessian.
        """
        from collections import defaultdict

        entries: dict = defaultdict(float)
        for coeff, v1, v2 in quad_terms:
            i, j = v1._col_idx, v2._col_idx
            if i == j:
                entries[(i, i)] += 2.0 * coeff
            else:
                row, col = min(i, j), max(i, j)
                entries[(row, col)] += coeff

        # Remove zero entries to keep the matrix sparse
        entries = {k: v for k, v in entries.items() if v != 0.0}

        n = len(self._vars)
        by_col: dict = defaultdict(list)
        for (row, col), val in entries.items():
            by_col[col].append((row, val))
        for col in by_col:
            by_col[col].sort()

        start = [0] * (n + 1)
        index: list = []
        value: list = []
        nnz = 0
        for col in range(n):
            start[col] = nnz
            for row, val in by_col.get(col, []):
                index.append(row)
                value.append(val)
                nnz += 1
        start[n] = nnz
        return start, index, value

    def setObjective(self, expr, sense: int | None = None) -> None:
        if sense is not None:
            self._obj_sense = sense

        if isinstance(expr, QuadExpr):
            linexpr = expr._lin_expr._copy()
            quad_terms = expr._quad_terms
        else:
            linexpr = _to_linexpr(expr)
            quad_terms = []

        # Reset every column cost to zero, then apply new costs
        for var in self._vars:
            self._h.changeColCost(var._col_idx, 0.0)
        for vid, (var, coeff) in linexpr._terms.items():
            self._h.changeColCost(var._col_idx, float(coeff))

        self._h.changeObjectiveOffset(linexpr.constant)
        self._obj_offset = linexpr.constant

        # Set or clear the Hessian
        hess = highspy.HighsHessian()
        hess.format_ = highspy.HessianFormat.kTriangular
        if quad_terms:
            start, index, value = self._build_hessian(quad_terms)
            hess.dim_ = len(self._vars)
            hess.start_ = start
            hess.index_ = index
            hess.value_ = value
            self._has_quad = True
        else:
            hess.dim_ = 0
            hess.start_ = []
            hess.index_ = []
            hess.value_ = []
            self._has_quad = False
        self._h.passHessian(hess)

        sense_enum = (
            highspy.ObjSense.kMinimize
            if self._obj_sense == GRB.MINIMIZE
            else highspy.ObjSense.kMaximize
        )
        self._h.changeObjectiveSense(sense_enum)

    def setAttr(self, attr: str, objs_or_value, values=None) -> None:
        """
        Set a model attribute or batch-set an attribute on a list of objects.

        Single-object form (model attributes):
            model.setAttr("ModelSense", GRB.MAXIMIZE)
            model.setAttr("ObjCon", 5.0)

        Batch form (variables or constraints):
            model.setAttr("LB", vars, [0.0, 1.0, 2.0])
            model.setAttr("RHS", constrs, [4.0, 6.0])
        """
        if values is None:
            # Model-level attribute
            value = objs_or_value
            if attr == "ModelSense":
                self._obj_sense = value
                sense_enum = (
                    highspy.ObjSense.kMinimize
                    if value == GRB.MINIMIZE
                    else highspy.ObjSense.kMaximize
                )
                self._h.changeObjectiveSense(sense_enum)
            elif attr == "ObjCon":
                self._obj_offset = float(value)
                self._h.changeObjectiveOffset(self._obj_offset)
            else:
                raise AttributeError(f"Cannot set model attribute '{attr}'")
        else:
            # Batch set: iterate over (obj, val) pairs and use property setters
            _setter_map = {
                "LB":             lambda o, v: setattr(o, "LB", v),
                "UB":             lambda o, v: setattr(o, "UB", v),
                "Obj":            lambda o, v: setattr(o, "Obj", v),
                "VType":          lambda o, v: setattr(o, "VType", v),
                "VarName":        lambda o, v: setattr(o, "VarName", v),
                "Start":          lambda o, v: setattr(o, "Start", v),
                "BranchPriority": lambda o, v: setattr(o, "BranchPriority", v),
                "RHS":            lambda o, v: setattr(o, "RHS", v),
                "Sense":          lambda o, v: setattr(o, "Sense", v),
                "ConstrName":     lambda o, v: setattr(o, "ConstrName", v),
            }
            fn = _setter_map.get(attr)
            if fn is None:
                raise AttributeError(
                    f"Cannot set attribute '{attr}' in batch mode"
                )
            for obj, val in zip(objs_or_value, values):
                fn(obj, val)

    def setParam(self, param_name: str, value) -> None:
        setattr(self.Params, param_name, value)

    # ------------------------------------------------------------------ #
    # Solve
    # ------------------------------------------------------------------ #

    def optimize(self, callback=None) -> None:
        self._solution = None
        self._cb_sol_count = 0

        # Apply MIP warm-start hints if any variable has Start set
        if any(s < GRB.INFINITY for s in self._var_starts):
            sol = highspy.HighsSolution()
            sol.col_value = [
                0.0 if s >= GRB.INFINITY else s
                for s in self._var_starts
            ]
            sol.value_valid = True
            self._h.setSolution(sol)

        if callback is None:
            self._h.run()
        else:
            # Outer loop: re-run whenever the user queued lazy constraints.
            while True:
                self._cb_lazy_queue = []
                self._cb_where = -1
                self._cb_data_out = None
                self._cb_data_in = None

                def _highs_cb(
                    cb_type, message, data_out, data_in, _user_data,
                    _self=self, _callback=callback,
                ):
                    ct = int(cb_type)
                    _self._cb_data_out = data_out
                    _self._cb_data_in = data_in
                    if ct == 3:  # kCallbackMipSolution
                        _self._cb_where = GRB.Callback.MIPSOL
                        _self._cb_sol_count += 1
                        _callback(GRB.Callback.MIPSOL)
                    # Clear references so terminate()/cbGet() outside the
                    # callback body are no-ops / raise correctly.
                    _self._cb_data_out = None
                    _self._cb_data_in = None
                    _self._cb_where = -1

                self._h.setCallback(_highs_cb, None)
                self._h.startCallback(self._h.cbMipSolution.callback_type)
                self._h.run()

                if not self._cb_lazy_queue:
                    break

                # Add queued lazy constraints to the model and re-solve.
                for tc in self._cb_lazy_queue:
                    self.addConstr(tc)

        self._solution = self._h.getSolution()
        self._status = _map_highs_status(self._h.getModelStatus())

    # ------------------------------------------------------------------ #
    # Callback helpers (valid only inside an optimize(callback) run)
    # ------------------------------------------------------------------ #

    def cbGet(self, what: int):
        """
        Return a progress metric for the current callback context.

        *what* must be one of the ``GRB.Callback.MIPSOL_*`` constants.
        Raises ``RuntimeError`` if called outside of a callback.
        Raises ``AttributeError`` for unknown *what* codes.
        """
        if self._cb_data_out is None:
            raise RuntimeError("cbGet() called outside of a callback context.")
        d = self._cb_data_out
        _map = {
            GRB.Callback.MIPSOL_OBJ:    lambda: float(d.objective_function_value),
            GRB.Callback.MIPSOL_OBJBST: lambda: float(d.mip_primal_bound),
            GRB.Callback.MIPSOL_OBJBND: lambda: float(d.mip_dual_bound),
            GRB.Callback.MIPSOL_NODCNT: lambda: int(d.mip_node_count),
            GRB.Callback.MIPSOL_SOLCNT: lambda: self._cb_sol_count,
        }
        fn = _map.get(what)
        if fn is None:
            raise AttributeError(f"Unknown callback what-code: {what!r}")
        return fn()

    def cbGetSolution(self, vars):
        """
        Return solution values for *vars* from the current MIPSOL callback.

        *vars* may be a single ``Var``, a list of ``Var`` objects, or a
        ``tupledict`` mapping keys to ``Var`` objects.  Returns a float,
        list of floats, or dict respectively.
        """
        if self._cb_data_out is None:
            raise RuntimeError("cbGetSolution() called outside of a callback context.")
        sol = self._cb_data_out.mip_solution
        if hasattr(vars, "_col_idx"):  # single Var
            return float(sol[vars._col_idx])
        if hasattr(vars, "values"):    # dict / tupledict
            return {k: float(sol[v._col_idx]) for k, v in vars.items()}
        return [float(sol[v._col_idx]) for v in vars]

    def cbLazy(self, constr) -> None:
        """
        Queue a lazy constraint to be added after the current solve completes.

        The constraint is a ``TempConstr`` produced by a comparison operator,
        e.g. ``model.cbLazy(x + y <= 10)``.  All queued constraints are added
        via ``model.addConstr()`` once ``h.run()`` returns, and the solver is
        re-started.  Only callable from within a ``MIPSOL`` callback.
        """
        if self._cb_where != GRB.Callback.MIPSOL:
            raise RuntimeError("cbLazy() can only be called from a MIPSOL callback.")
        self._cb_lazy_queue.append(constr)

    def terminate(self) -> None:
        """
        Request early termination of the current solve.

        When called from inside a ``MIPSOL`` callback, the solver stops after
        the callback returns and ``optimize()`` returns with the best solution
        found so far.  Calling ``terminate()`` outside of a callback is a
        no-op.
        """
        if self._cb_data_in is not None:
            self._cb_data_in.user_interrupt = True

    def remove(self, items) -> None:
        """
        Remove one or more variables or constraints from the model.

        Accepts a single Var, a single Constr, or a list containing any mix
        of the two.  After removal:

        * The removed objects are marked invalid — accessing their attributes
          raises RuntimeError.
        * The col/row indices of all surviving objects are updated to match
          the new HiGHS model layout.
        * The current solution is cleared (call optimize() again to re-solve).
        """
        if isinstance(items, (Var, Constr, SOS, GenConstr)):
            items = [items]

        # Expand SOS objects into their helper vars and constrs for deletion.
        sos_to_remove: list = [s for s in items if isinstance(s, SOS)]
        if sos_to_remove:
            extra_vars: list = []
            extra_constrs: list = []
            for sos in sos_to_remove:
                extra_vars.extend(sos._helper_vars)
                extra_constrs.extend(sos._helper_constrs)
            items = (
                [x for x in items if not isinstance(x, SOS)]
                + extra_vars
                + extra_constrs
            )

        # Expand GenConstr objects into their helper vars and constrs.
        gc_to_remove: list = [g for g in items if isinstance(g, GenConstr)]
        if gc_to_remove:
            extra_vars2: list = []
            extra_constrs2: list = []
            for gc in gc_to_remove:
                extra_vars2.extend(gc._helper_vars)
                extra_constrs2.extend(gc._helper_constrs)
            items = (
                [x for x in items if not isinstance(x, GenConstr)]
                + extra_vars2
                + extra_constrs2
            )

        vars_to_remove = [v for v in items if isinstance(v, Var)]
        constrs_to_remove = [c for c in items if isinstance(c, Constr)]

        # ---- Remove variables ----------------------------------------- #
        if vars_to_remove:
            cols = sorted(v._col_idx for v in vars_to_remove)
            self._h.deleteVars(len(cols), cols)

            cols_set = set(cols)
            self._vars               = [v  for v  in self._vars               if v._col_idx not in cols_set]
            self._var_lbs            = [lb for i, lb in enumerate(self._var_lbs)            if i not in cols_set]
            self._var_ubs            = [ub for i, ub in enumerate(self._var_ubs)            if i not in cols_set]
            self._var_objs           = [o  for i, o  in enumerate(self._var_objs)           if i not in cols_set]
            self._var_vtypes         = [t  for i, t  in enumerate(self._var_vtypes)         if i not in cols_set]
            self._var_starts         = [s  for i, s  in enumerate(self._var_starts)         if i not in cols_set]
            self._var_branch_priorities = [p for i, p in enumerate(self._var_branch_priorities) if i not in cols_set]

            # Recompute col indices: each remaining var shifts down by the
            # number of deleted indices that were strictly less than its index.
            for var in self._vars:
                var._col_idx -= bisect.bisect_left(cols, var._col_idx)

            for var in vars_to_remove:
                var._col_idx = -1

        # ---- Remove constraints --------------------------------------- #
        if constrs_to_remove:
            rows = sorted(c._row_idx for c in constrs_to_remove)
            self._h.deleteRows(len(rows), rows)

            rows_set = set(rows)
            self._constrs = [c for c in self._constrs if c._row_idx not in rows_set]

            for constr in self._constrs:
                constr._row_idx -= bisect.bisect_left(rows, constr._row_idx)

            for constr in constrs_to_remove:
                constr._row_idx = -1

        # Mark removed SOS objects as invalid and remove from list.
        for sos in sos_to_remove:
            sos._removed = True
            if sos in self._sos_list:
                self._sos_list.remove(sos)

        # Mark removed GenConstr objects as invalid and remove from list.
        for gc in gc_to_remove:
            gc._removed = True
            if gc in self._gen_constrs:
                self._gen_constrs.remove(gc)

        # Clear stale solution state.
        self._solution = None
        self._status = GRB.LOADED

    def copy(self) -> Model:
        """
        Return a new Model that is a structural copy of this one.

        The copy has identical variables, constraints, objective coefficients,
        objective sense, and variable types.  It starts in the unsolved state —
        solution values and dual information are not transferred.  The two
        models are fully independent: changes to one do not affect the other.
        """
        new_model = Model(self._name)

        # Transfer the full HiGHS model (vars, rows, costs, integrality,
        # objective sense, offset, and Hessian if QP) in one call.
        new_model._h.passModel(self._h.getModel())

        # Copy Python-side metadata directly from source (authoritative).
        new_model._var_lbs               = list(self._var_lbs)
        new_model._var_ubs               = list(self._var_ubs)
        new_model._var_objs              = list(self._var_objs)
        new_model._var_vtypes            = list(self._var_vtypes)
        new_model._var_starts            = list(self._var_starts)
        new_model._var_branch_priorities = list(self._var_branch_priorities)
        new_model._obj_offset            = self._obj_offset
        new_model._has_quad              = self._has_quad

        # Create new Var objects that point into the new HiGHS instance.
        new_model._vars = [
            Var(v._col_idx, new_model, name=v._name)
            for v in self._vars
        ]

        # Create new Constr objects that point into the new HiGHS instance.
        new_model._constrs = [
            Constr(c._row_idx, new_model, name=c._name, sense=c._sense, rhs=c._rhs)
            for c in self._constrs
        ]

        new_model._obj_sense = self._obj_sense

        # Reconstruct SOS objects, remapping vars/constrs by index.
        new_model._sos_list = []
        for sos in self._sos_list:
            new_vars = [new_model._vars[v._col_idx] for v in sos._vars]
            new_helpers = [new_model._vars[v._col_idx] for v in sos._helper_vars]
            new_hconstrs = [new_model._constrs[c._row_idx] for c in sos._helper_constrs]
            new_sos = SOS(new_model, sos._sostype, new_vars, list(sos._weights), name=sos._name)
            new_sos._helper_vars = new_helpers
            new_sos._helper_constrs = new_hconstrs
            new_model._sos_list.append(new_sos)

        # Reconstruct GenConstr objects.
        new_model._gen_constrs = []
        for gc in self._gen_constrs:
            new_gc = GenConstr(new_model, gc._type, name=gc._name)
            new_gc._helper_vars = [new_model._vars[v._col_idx] for v in gc._helper_vars]
            new_gc._helper_constrs = [
                new_model._constrs[c._row_idx] for c in gc._helper_constrs
            ]
            new_model._gen_constrs.append(new_gc)

        return new_model

    def update(self) -> None:
        """No-op: all changes are applied immediately in this wrapper."""

    def reset(self) -> None:
        """Clear the current solution without re-building the model."""
        self._solution = None
        self._status = GRB.LOADED

    def computeIIS(self) -> None:
        """Compute an Irreducible Inconsistent Subsystem (best-effort)."""
        self._h.run()

    # ------------------------------------------------------------------ #
    # Solution properties
    # ------------------------------------------------------------------ #

    @property
    def ObjVal(self) -> float:
        _, val = self._h.getInfoValue("objective_function_value")
        return float(val)

    @property
    def ObjBound(self) -> float:
        _, val = self._h.getInfoValue("mip_dual_bound")
        return float(val)

    @property
    def MIPGap(self) -> float:
        _, val = self._h.getInfoValue("mip_gap")
        return float(val)

    @property
    def Status(self) -> int:
        return self._status

    @property
    def Runtime(self) -> float:
        _, val = self._h.getInfoValue("run_time")
        return float(val)

    @property
    def NumVars(self) -> int:
        return len(self._vars)

    @property
    def NumConstrs(self) -> int:
        return len(self._constrs)

    @property
    def ModelName(self) -> str:
        return self._name

    @ModelName.setter
    def ModelName(self, value: str) -> None:
        self._name = value

    @property
    def ModelSense(self) -> int:
        return self._obj_sense

    @ModelSense.setter
    def ModelSense(self, value: int) -> None:
        self.setAttr("ModelSense", value)

    @property
    def ObjCon(self) -> float:
        """Objective constant (offset added to the objective value)."""
        return self._obj_offset

    @ObjCon.setter
    def ObjCon(self, value: float) -> None:
        self.setAttr("ObjCon", value)

    # ------------------------------------------------------------------ #
    # Retrieval helpers
    # ------------------------------------------------------------------ #

    def getVars(self) -> list[Var]:
        return list(self._vars)

    def getConstrs(self) -> list[Constr]:
        return list(self._constrs)

    def getVarByName(self, name: str) -> Var | None:
        for v in self._vars:
            if v._name == name:
                return v
        return None

    def getConstrByName(self, name: str) -> Constr | None:
        for c in self._constrs:
            if c._name == name:
                return c
        return None

    def getAttr(self, attr: str, objs=None):
        if objs is None:
            return getattr(self, attr)
        _getter_map = {
            # Variable attributes
            "X":             lambda o: o.X,
            "RC":            lambda o: o.RC,
            "LB":            lambda o: o.LB,
            "UB":            lambda o: o.UB,
            "Obj":           lambda o: o.Obj,
            "VarName":       lambda o: o.VarName,
            "VType":         lambda o: o.VType,
            "Start":         lambda o: o.Start,
            "BranchPriority":lambda o: o.BranchPriority,
            # Constraint attributes
            "Pi":            lambda o: o.Pi,
            "Slack":         lambda o: o.Slack,
            "ConstrName":    lambda o: o.ConstrName,
            "RHS":           lambda o: o.RHS,
            "Sense":         lambda o: o.Sense,
        }
        fn = _getter_map.get(attr)
        if fn is None:
            raise AttributeError(f"Unknown attribute '{attr}'")
        return [fn(o) for o in objs]

    # ------------------------------------------------------------------ #
    # I/O
    # ------------------------------------------------------------------ #

    def write(self, filename: str) -> None:
        self._h.writeModel(filename)

    def read(self, filename: str) -> None:
        """
        Load a model from a file (.lp, .mps, or any HiGHS-supported format).

        Replaces any existing model state.  After this call, ``getVars()`` and
        ``getConstrs()`` return wrapper objects backed by the loaded columns and
        rows, and the model is ready to optimize.
        """
        self._h.readModel(filename)
        lp = self._h.getLp()

        # ---- Variable metadata ---------------------------------------- #
        self._var_lbs  = list(lp.col_lower_)
        self._var_ubs  = list(lp.col_upper_)
        self._var_objs = list(lp.col_cost_)

        # Map integrality; kInteger with bounds [0, 1] → BINARY
        integrality = list(lp.integrality_)
        self._var_vtypes = []
        for i in range(lp.num_col_):
            if i < len(integrality) and integrality[i] == highspy.HighsVarType.kInteger:
                if lp.col_lower_[i] == 0.0 and lp.col_upper_[i] == 1.0:
                    self._var_vtypes.append(GRB.BINARY)
                else:
                    self._var_vtypes.append(GRB.INTEGER)
            else:
                self._var_vtypes.append(GRB.CONTINUOUS)

        col_names = list(lp.col_names_)
        self._vars = [
            Var(i, self, name=col_names[i] if i < len(col_names) else "")
            for i in range(lp.num_col_)
        ]
        self._var_starts            = [GRB.UNDEFINED] * lp.num_col_
        self._var_branch_priorities = [0] * lp.num_col_

        # ---- Constraint metadata --------------------------------------- #
        # HiGHS stores constraints as double-sided row bounds; infer sense.
        row_names = list(lp.row_names_)
        self._constrs = []
        for i in range(lp.num_row_):
            row_lb = float(lp.row_lower_[i])
            row_ub = float(lp.row_upper_[i])
            lb_inf = row_lb <= -_INF / 2
            ub_inf = row_ub >= _INF / 2
            if lb_inf and not ub_inf:
                sense, rhs = GRB.LESS_EQUAL, row_ub
            elif not lb_inf and ub_inf:
                sense, rhs = GRB.GREATER_EQUAL, row_lb
            else:
                sense, rhs = GRB.EQUAL, row_lb
            name = row_names[i] if i < len(row_names) else ""
            self._constrs.append(
                Constr(i, self, name=name, sense=sense, rhs=rhs)
            )

        # ---- Objective sense and offset ------------------------------- #
        self._obj_sense = (
            GRB.MAXIMIZE
            if lp.sense_ == highspy.ObjSense.kMaximize
            else GRB.MINIMIZE
        )
        self._obj_offset = float(lp.offset_)

        # ---- Quadratic objective (Hessian) ---------------------------- #
        hessian = self._h.getModel().hessian_
        self._has_quad = hessian.dim_ > 0 and len(hessian.value_) > 0

        # ---- Clear stale solution ------------------------------------- #
        self._solution = None
        self._status = GRB.LOADED

    # ------------------------------------------------------------------ #
    # Context manager / cleanup
    # ------------------------------------------------------------------ #

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.dispose()

    def dispose(self) -> None:
        """Release HiGHS resources."""
        # highspy cleans up via GC; nothing explicit needed

    def __repr__(self) -> str:
        return (
            f"<grbcompat.Model '{self._name}' "
            f"vars={self.NumVars} constrs={self.NumConstrs} "
            f"status={self._status}>"
        )
