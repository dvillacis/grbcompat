# Roadmap

This document tracks planned work for `grb-highs-wrapper`, ordered by impact and implementation difficulty. Each milestone is designed to ship independently.

---

## Current state — v0.4

Full drop-in replacement for the core gurobipy LP/MIP/QP API backed by HiGHS:

- `Model`, `Var`, `Constr`, `SOS`, `GenConstr`, `LinExpr`, `QuadExpr`, `tupledict`, `GRB` constants, `Params`
- `addVar` / `addVars` / `addConstr` / `addConstrs` / `setObjective` / `optimize`
- Dual values, slacks, reduced costs after solve
- `install()` for zero-code-change drop-in use
- `Model.remove()` — single and batch deletion of variables, constraints, SOS, and GenConstr
- `Model.copy()` — full structural copy; independent solve and modification (Hessian included)
- `Model.read(filename)` — load LP/MPS files; full wrapper state reconstructed; `_has_quad` set from Hessian
- `Model.write(filename)` — LP and MPS round-trips for both LP and QP models verified
- `Model.addSOS()` — SOS1 via big-M linearization; SOS2 raises `NotImplementedError`
- `Model.addGenConstrIndicator()` — if/then constraints via big-M
- `Model.addGenConstrAbs()` — absolute value constraints via big-M + binary
- `Model.addGenConstrMin()` / `addGenConstrMax()` — min/max constraints via big-M + binary selection
- `QuadExpr` — full arithmetic (`+`, `-`, `*`, `/`, scalar scaling, `LinExpr * Var`)
- `setObjective(QuadExpr)` — convex QP via HiGHS `passHessian()`; upper-triangular CSC Hessian built automatically
- `addQConstr` — raises `NotImplementedError` (HiGHS supports quadratic objectives only)
- `model.optimize(callback)` — MIPSOL callbacks with `cbGet`, `cbGetSolution`, `cbLazy`, `terminate`
- `GRB.Callback` constants (`MIPSOL`, `MIPNODE`, `MIP`, `SIMPLEX`, `BARRIER`, `MIPSOL_OBJ`, ...)
- `Params.LazyConstraints` — accepted as no-op (lazy constraints are handled in Python outer loop)
- 807 tests; cross-solver comparison tests against real Gurobi

Known gaps that block real-world usage are captured below.

---

## v0.2 — Critical API gaps

*Target: unblock the largest fraction of gurobipy code that currently fails.*

### ~~`Model.remove()`~~ ✓ shipped in v0.1.1

`model.remove(var)`, `model.remove(constr)`, and `model.remove(list)` are fully implemented. Surviving `Var` and `Constr` objects have their indices recomputed in O(n log n) using `bisect`. Removed objects raise `RuntimeError` on any attribute access. Covered by 39 dedicated tests.

---

### ~~`Model.copy()`~~ ✓ shipped in v0.1.2

`model.copy()` returns a fully independent `Model` with identical variables, constraints, objective sense/offset, variable types, and bounds. Implemented via `passModel(getLp())` to transfer the HiGHS model, with new Python `Var`/`Constr` objects pointing into the new instance. Solution state is not copied — the copy starts unsolved. Covered by 40 dedicated tests.

---

### ~~`Model.read(filename)`~~ ✓ shipped in v0.1.3

`model.read(filename)` loads any HiGHS-supported format (`.lp`, `.mps`) via `readModel`, then rebuilds all Python wrapper state from `getLp()`: variable bounds, objective coefficients, integrality (with BINARY detection from `kInteger` + `[0,1]` bounds), constraint sense/RHS (inferred from row bounds), objective sense, and — from v0.4 — `_has_quad` (detected from the loaded Hessian via `getModel().hessian_`). Replaces any existing state. Covered by 43 dedicated tests. Note: HiGHS auto-generates column/row names (`c0`, `c1`, ..., `r0`, `r1`, ...) since user names are stored only in the Python wrapper and are not written to files.

---

### ~~SOS constraints~~ ✓ shipped in v0.2

`model.addSOS(type, vars, weights)` is fully implemented for SOS1.  SOS2 raises `NotImplementedError` (HiGHS has no native SOS2 support and the linearized reformulation is impractical without it).

SOS1 is implemented via big-M linearization: for each member variable `x_i` with finite upper bound `UB_i`, a binary indicator `b_i` is introduced with constraint `x_i <= UB_i * b_i`, and the global constraint `sum(b_i) <= 1` enforces that at most one variable can be nonzero.  All member variables must have finite upper bounds.  `GRB.SOS_TYPE1 = 1` and `GRB.SOS_TYPE2 = 2` added to `_constants.py`.  `Model.remove()` and `Model.copy()` both handle SOS objects correctly.  Covered by 42 dedicated tests.

---

### ~~General constraints~~ ✓ shipped in v0.2

All four general constraint types are implemented via big-M linearization and exposed through a `GenConstr` wrapper object:

- **`addGenConstrIndicator(binvar, binval, constr, name="")`** — `if binvar == binval then lhs sense rhs`. One helper constraint per inequality sense (two for equality). Big-M is computed from variable bounds in the LHS expression; falls back to 1e6 if bounds are infinite.
- **`addGenConstrAbs(resvar, argvar, name="")`** — `resvar = |argvar|`. One binary helper variable and four linear constraints enforce sign-consistency and equality.
- **`addGenConstrMin(resvar, argvars, constant=∞, name="")`** — `resvar = min(argvars [, constant])`. One binary per candidate + `sum = 1` + upper/lower bound constraints.
- **`addGenConstrMax(resvar, argvars, constant=-∞, name="")`** — symmetric to min.

`Model.remove()` and `Model.copy()` both handle `GenConstr` objects correctly. Covered by 66 dedicated tests.

---

### ~~Batch `getAttr` / `setAttr` completeness~~ ✓ shipped in v0.2

`model.getAttr(attr, objs)` now covers all variable attributes (`X`, `RC`, `LB`, `UB`, `Obj`, `VarName`, `VType`, `Start`, `BranchPriority`) and all constraint attributes (`Pi`, `Slack`, `RHS`, `Sense`, `ConstrName`).

`model.setAttr(attr, objs, values)` now supports batch writes for the same attribute set via property setters. `model.setAttr(attr, value)` still handles model-level attributes (`ModelSense`, `ObjCon`).

Additional completeness improvements shipped alongside:
- `GRB.UNDEFINED = 1e101` — sentinel for unset MIP start hints
- `Var.Start` / `Var.BranchPriority` — new per-variable attributes; `Start` hints are applied to HiGHS via `HighsSolution.setSolution()` before each solve
- `Constr.Sense` setter — change constraint sense in-place; updates HiGHS row bounds immediately
- `Model.ObjCon` property — read/write the objective constant offset; synced from `setObjective()`, `setAttr("ObjCon", ...)`, and `read()`

Covered by 69 dedicated tests.

---

## ~~v0.3 — Quadratic programming~~ ✓ shipped in v0.3

`QuadExpr` supports full arithmetic (`+`, `-`, `*` for Var×Var, scalar×QuadExpr, QuadExpr+LinExpr, LinExpr*Var, etc.), `size()`, `getCoeff(i)`, `getVar1(i)`, `getVar2(i)`, `getLinExpr()`, `getValue()`, and comparison operators returning `TempQConstr`.

`setObjective(quad_expr, sense)` converts the quadratic part into an upper-triangular CSC Hessian (Q, where HiGHS computes `0.5 * x'Qx`) via `passHessian()`. Diagonal terms `k*x_i^2` → `Q[i,i]=2k`; cross terms `k*x_i*x_j` → `Q[min,max]=k`. Calling `setObjective` with a plain `LinExpr` clears the Hessian. `copy()` uses `getModel()` (instead of `getLp()`) to transfer the Hessian to the copy.

`addQConstr` raises `NotImplementedError` — HiGHS supports quadratic objectives only, not quadratic constraints.

`model.write()` round-trips for QP models verified for both `.lp` (uses `[...]/2` notation) and `.mps` (uses `QUADOBJ` section) formats. `model.read()` correctly detects and sets `_has_quad` from the loaded Hessian. Covered by 53 dedicated tests (40 original + 13 write/round-trip).

---

## ~~v0.4 — Callbacks~~ ✓ shipped in v0.4

`model.optimize(callback)` fires the user callback with `where = GRB.Callback.MIPSOL` whenever HiGHS finds a new MIP feasible solution (`kCallbackMipSolution`).

**Supported in callbacks:**
- `model.cbGetSolution(var_or_list_or_tupledict)` — current solution values from `data_out.mip_solution`
- `model.cbGet(GRB.Callback.MIPSOL_OBJ/OBJBST/OBJBND/NODCNT/SOLCNT)` — progress metrics
- `model.cbLazy(constr)` — queues a lazy `TempConstr`; after `h.run()` the wrapper calls `addConstr()` for each queued constraint and re-runs (outer-loop pattern)
- `model.terminate()` — sets `data_in.user_interrupt = True`, stopping the solver after the callback returns
- `Params.LazyConstraints = 1` — accepted as no-op

**HiGHS limitations (documented):**
- `MIPNODE`, `MIP`, `SIMPLEX`, `BARRIER` where codes are defined in `GRB.Callback` but HiGHS does not fire them — those branches in user callbacks will simply never execute.
- Adding rows mid-solve (`h.addRow()` in callback) causes `kSolveError`; hence the outer-loop pattern for lazy constraints.
- `terminate()` only takes effect when called from within a MIPSOL callback (requires live `data_in` object).

Covered by 38 dedicated tests.

---

## v0.5 — Solver plugin system

*Target: make the backend swappable so the wrapper becomes a solver-agnostic gurobipy execution layer, not just a Gurobi-to-HiGHS bridge.*

### Backend interface

Define an abstract `Backend` protocol with the operations `Model` currently calls directly on `highspy.Highs`:

```
Backend.add_var(lb, ub, obj, vtype, name) -> int
Backend.add_row(coeffs, lb, ub, name)     -> int
Backend.set_objective(coeffs, offset, sense)
Backend.solve() -> status
Backend.get_solution() -> SolutionData
Backend.set_option(name, value)
...
```

`_model.py` talks only to this interface. `HiGHSBackend` implements it (wrapping the current `highspy` calls). New backends implement the same interface.

### Bundled backends

| Backend | Solver | License | Notes |
|---|---|---|---|
| `HiGHSBackend` | HiGHS | MIT | Default; already implemented |
| `CBCBackend` | CBC (via `cylp` or `python-mip`) | EPL | Pure open-source alternative |
| `SCIPBackend` | SCIP (via `pyscipopt`) | Academic/commercial | Better MIP performance on some instances |
| `GurobiBackend` | Gurobi | Commercial | Enables side-by-side benchmarking from a single API |

### Backend selection

```python
import grb_highs_wrapper as gp

m = gp.Model("lp", backend="highs")    # default
m = gp.Model("lp", backend="scip")
m = gp.Model("lp", backend="gurobi")
```

The `backend` parameter is the primary motivator here: it answers the open question about elegant solver selection without requiring different import paths or a separate API.

---

## v0.6 — Performance and correctness hardening

### Benchmarking suite

Publish a reproducible benchmark comparing HiGHS and Gurobi on:

- LP: NETLIB instances (155 standard problems)
- MIP: MIPLIB 2017 easy/medium instances
- Metrics: solve time, objective value gap, number of B&B nodes

Results published as a table in the repository and updated on each release. This answers the first question every prospective user asks.

### Numerical edge cases

- Unbounded variables (`lb = -GRB.INFINITY`)
- Degenerate problems (multiple optimal bases)
- Near-infeasible problems (presolve sensitivity)
- Very large / very small coefficients (scaling)

### ~~Warm starting~~ ✓ shipped in v0.2

`Var.Start` attribute hints the MIP solver with an incumbent solution via `HighsSolution.setSolution()` before each `h.run()`. `GRB.UNDEFINED` sentinel marks unset hints.

---

## v1.0 — Stable public API

*Target: declare the public interface stable and commit to semantic versioning.*

- All items from v0.2–v0.5 complete.
- Public API documented with type annotations throughout.
- No breaking changes without a major version bump from this point.
- `CHANGELOG.md` maintained from this release forward.

### What v1.0 will not include

These are explicitly out of scope indefinitely due to HiGHS limitations or complexity:

- **Multi-objective optimisation** (`model.setObjectiveN`) — HiGHS has no native multi-objective support.
- **Scenario analysis** (`model.addScenario`) — Gurobi-specific feature.
- **Network simplex / network flow specialisation** — HiGHS uses general simplex; no network-specific API.
- **Distributed MIP** — requires solver-level support.

---

## Infrastructure (ongoing)

These are not tied to a release milestone but should be completed early.

### GitHub Actions CI

```
.github/workflows/ci.yml
  - matrix: python [3.11, 3.12, 3.13] × os [ubuntu, macos, windows]
  - steps: install deps → run pytest → upload coverage
  - trigger: push to main, all PRs
```

### Automated release workflow

On tag push (`v*`):
1. Run full test suite.
2. `uv build` → produces wheel + sdist.
3. `twine upload` to PyPI using a stored API token secret.

### Coverage gate

Enforce ≥ 90% line coverage in CI. Any PR that drops coverage must include an explanation.

### `CHANGELOG.md`

Follow [Keep a Changelog](https://keepachangelog.com) format. Each release documents added, changed, fixed, and removed items.

---

## Contribution priorities

If you want to contribute, the highest-leverage items in order are:

1. **Fix packaging** — `pyproject.toml` points to `src/grbcompat` but source is in `src/grb_highs_wrapper`; the built wheel is empty until this is corrected
2. **GitHub Actions CI** — infrastructure that builds confidence for every future contribution
3. **Benchmarking suite** — establishes credibility and sets a performance baseline (v0.6)
4. **Solver plugin system** — makes the backend swappable (v0.5)
5. **CHANGELOG.md** — required before any PyPI release
