# grbcompat

[![PyPI](https://img.shields.io/pypi/v/grbcompat)](https://pypi.org/project/grbcompat/)
[![CI](https://github.com/dvillacis/grbcompat/actions/workflows/ci.yml/badge.svg)](https://github.com/dvillacis/grbcompat/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A Python package that lets you run [gurobipy](https://www.gurobi.com/documentation/current/refman/py_python_api_details.html)-based optimization models on the free, open-source [HiGHS](https://highs.dev) solver — with **no changes to your existing code**.

## Why

Gurobi is a best-in-class commercial solver, but it requires a licence. HiGHS is a high-quality open-source LP/MIP solver. This wrapper translates the gurobipy API into HiGHS calls at runtime, so you can:

- Run models on HiGHS during development without a Gurobi licence.
- Switch freely between solvers to compare results or costs.
- Keep a single codebase that works with either solver.

## Requirements

- Python ≥ 3.11
- [`highspy`](https://pypi.org/project/highspy/) ≥ 1.7 (installed automatically)
- A Gurobi licence is **not** required to use this wrapper.

## Installation

```bash
pip install grbcompat
```

Or, to work from source:

```bash
git clone <repo-url>
cd grbcompat
uv sync --dev
```

---

## Usage

### Pattern A — Drop-in replacement (one extra line)

Add a single `install()` call **before** any `import gurobipy` statement. Every subsequent `import gurobipy` in that process resolves to the HiGHS-backed wrapper. Your existing code stays untouched.

```python
import grbcompat
grbcompat.install()          # ← the only change

import gurobipy as gp                # now backed by HiGHS
from gurobipy import GRB

m = gp.Model("my_model")
x = m.addVar(lb=0, name="x")
y = m.addVar(lb=0, name="y")

m.setObjective(x + 2 * y, GRB.MINIMIZE)
m.addConstr(x + y >= 1, "demand")
m.optimize()

print(f"x={x.X:.4f}  y={y.X:.4f}  obj={m.ObjVal:.4f}")
```

### Pattern B — Side-by-side solvers

Import the wrapper directly under its own name. Both solver namespaces are fully independent and can be used in the same script simultaneously.

```python
import grbcompat as highs    # backed by HiGHS (free)
import gurobipy as gp                # backed by Gurobi (licensed)

from grbcompat import GRB as HGRB

# Build the same LP with both solvers
def build(Mod, G):
    m = Mod()
    x = m.addVar(lb=0.0, name="x")
    y = m.addVar(lb=0.0, name="y")
    m.addConstr(x + y >= 4, "demand")
    m.setObjective(2 * x + 3 * y, G.MINIMIZE)
    m.optimize()
    return m, x, y

m_h, x_h, y_h = build(highs.Model, HGRB)
m_g, x_g, y_g = build(gp.Model, gp.GRB)

print(f"HiGHS:  obj={m_h.ObjVal:.4f}  x={x_h.X:.4f}  y={y_h.X:.4f}")
print(f"Gurobi: obj={m_g.ObjVal:.4f}  x={x_g.X:.4f}  y={y_g.X:.4f}")
```

---

## Supported gurobipy API

### `Model`

| Method / Property | Notes |
|---|---|
| `addVar(lb, ub, obj, vtype, name)` | All types: `CONTINUOUS`, `INTEGER`, `BINARY` |
| `addVars(*indices, lb, ub, obj, vtype, name)` | Scalar, list, range, or cartesian-product indices; `lb`/`ub` can be dicts |
| `addConstr(lhs, sense, rhs, name)` | TempConstr from `<=`/`>=`/`==`, or explicit sense/rhs |
| `addConstrs(generator, name)` | Returns a `tupledict` |
| `addLConstr(...)` | Alias for `addConstr` |
| `setObjective(expr, sense)` | `GRB.MINIMIZE` / `GRB.MAXIMIZE` |
| `optimize()` | |
| `update()` | No-op (changes applied immediately) |
| `reset()` | Clears solution without rebuilding the model |
| `write(filename)` | Writes `.lp`, `.mps`, or other HiGHS-supported formats |
| `setParam(name, value)` | Alias for `model.Params.<name> = value` |
| `setAttr(attr, value)` | Supports `ModelSense`, `ObjCon` |
| `getAttr(attr, objects)` | Batch-reads `X`, `Pi`, `Slack`, `RC`, `LB`, `UB`, … |
| `getVars()` / `getConstrs()` | |
| `getVarByName(name)` / `getConstrByName(name)` | |
| `ObjVal`, `ObjBound`, `MIPGap` | After `optimize()` |
| `Status`, `Runtime` | |
| `NumVars`, `NumConstrs` | |
| `ModelName`, `ModelSense` | Readable and writable |
| Context manager (`with Model() as m`) | |

### `Var`

| Attribute / Property | Notes |
|---|---|
| `X` | Primal value (after `optimize()`) |
| `RC` | Reduced cost |
| `LB`, `UB`, `Obj`, `VType`, `VarName` | Readable and writable; changes take effect immediately |
| `+`, `-`, `*`, `/`, unary `-` | Returns `LinExpr` |
| `<=`, `>=`, `==` | Returns `TempConstr` for use in `addConstr` |

### `Constr`

| Attribute / Property | Notes |
|---|---|
| `Pi` | Dual value / shadow price (after `optimize()`) |
| `Slack` | Constraint slack (after `optimize()`) |
| `ConstrName`, `Sense`, `RHS` | `RHS` is writable; change takes effect at next `optimize()` |

### `LinExpr`

Full arithmetic support: `+`, `-`, `*`, `/`, unary `-`, `sum()` built-in, `==`/`<=`/`>=` to produce constraints.  
Methods: `size()`, `getCoeff(i)`, `getVar(i)`, `getConstant()`, `add()`, `addTerms()`, `getValue()`.

### `tupledict`

Returned by `addVars` and `addConstrs`. Supports `select(*pattern)`, `sum(*pattern)`, `prod(coeff_dict, *pattern)`.

### `GRB` constants

```python
GRB.MINIMIZE / GRB.MAXIMIZE
GRB.CONTINUOUS / GRB.INTEGER / GRB.BINARY / GRB.SEMICONT / GRB.SEMIINT
GRB.LESS_EQUAL / GRB.GREATER_EQUAL / GRB.EQUAL
GRB.INFINITY
GRB.OPTIMAL / GRB.INFEASIBLE / GRB.UNBOUNDED / GRB.INF_OR_UNBD
GRB.TIME_LIMIT / GRB.ITERATION_LIMIT / GRB.SOLUTION_LIMIT / …
GRB.Status.*   # mirrors the top-level codes
GRB.Attr.*     # attribute name strings
```

### `Params`

Set solver parameters via attribute assignment. gurobipy names are translated to their HiGHS equivalents automatically.

| gurobipy name | HiGHS option | Type |
|---|---|---|
| `TimeLimit` | `time_limit` | float |
| `MIPGap` | `mip_rel_gap` | float |
| `MIPGapAbs` | `mip_abs_gap` | float |
| `FeasibilityTol` | `primal_feasibility_tolerance` | float |
| `OptimalityTol` | `dual_feasibility_tolerance` | float |
| `IntFeasTol` | `mip_feasibility_tolerance` | float |
| `OutputFlag` | `output_flag` | bool |
| `Threads` | `threads` | int |
| `Seed` | `random_seed` | int |
| `NodeLimit` | `mip_max_nodes` | int |
| `IterationLimit` | `simplex_iteration_limit` | int |

Unknown parameter names are forwarded directly to HiGHS.

### Compatibility stubs

`GurobiError`, `Env`, `disposeDefaultEnv` — accepted and silently ignored so that licence-management code in existing scripts does not break.

### Module-level functions

```python
grbcompat.quicksum(iterable)      # → LinExpr
grbcompat.multidict(data)         # → [keys, dict1, dict2, …]
grbcompat.install()               # patches sys.modules['gurobipy']
```

---

## Running the tests

```bash
uv sync --dev
uv run pytest
```

The test suite has **388 tests** across 10 files:

| File | Tests | Coverage |
|---|---|---|
| `test_constants.py` | 28 | All `GRB` constants and status codes |
| `test_expr.py` | 47 | `LinExpr` arithmetic, merging, `TempConstr` |
| `test_var.py` | 52 | All `Var` properties, setters, operators |
| `test_constr.py` | 22 | `Constr` metadata, duals, slacks, `RHS` setter |
| `test_tupledict.py` | 31 | `select`, `sum`, `prod`, `addVars` integration |
| `test_params.py` | 16 | All mapped params, unknown-param error |
| `test_model_core.py` | 81 | `addVar/Vars`, `addConstr/Constrs`, `setObjective`, I/O |
| `test_model_lp.py` | 32 | LP correctness, duals, re-optimization |
| `test_model_mip.py` | 15 | Binary, integer, mixed-integer programs |
| `test_api_compat.py` | 55 | `install()` isolation, gurobipy usage patterns |
| `test_gurobi_comparison.py` | 29 | Cross-solver comparison (skipped without Gurobi licence) |

### Cross-solver comparison tests

If you have a valid Gurobi licence, the comparison tests run automatically and verify that HiGHS and Gurobi produce identical results (within `1e-4` tolerance) for LP objectives, primal values, dual values, slacks, solve status, and MIP objectives:

```bash
pytest tests/test_gurobi_comparison.py -v
```

Without a licence they are skipped with a clear message — no configuration needed.

---

## Limitations

- **Quadratic objectives and constraints** (`QuadExpr`, `QConstr`) are not supported. HiGHS does support QP but the translation layer is not yet implemented.
- **`Model.remove()`** (removing individual variables or constraints) is not implemented.
- **Callbacks** (`Model.optimize(callback)`) are not supported.
- **Multi-objective** and **scenario** features are not supported.
- HiGHS and Gurobi may return different optimal bases for degenerate problems; primal solution values can differ while objective values agree.

---

## How it works

When you call `install()`, the package registers itself under `sys.modules['gurobipy']`. Subsequent `import gurobipy` statements in the same process return the wrapper instead of the real package.

Internally, each `Model` creates a `highspy.Highs` instance. Variable and constraint operations translate directly to HiGHS column/row operations applied immediately (no lazy batching). `model.update()` is a no-op.

The solver-status integer codes (`GRB.OPTIMAL = 2`, etc.) match gurobipy's values exactly, so any code that branches on `model.Status` works correctly with either solver.

---

## License

MIT
