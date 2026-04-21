"""
grbcompat usage examples.

Two usage patterns are demonstrated:

Pattern A – Drop-in replacement (no changes to existing code)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Call ``grbcompat.install()`` once before any ``import gurobipy``
statement. From that point on, ``gurobipy`` resolves to this wrapper and
every existing gurobipy script runs unchanged.

Pattern B – Side-by-side (both solvers in the same script)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Import the wrapper under its own name (``import grbcompat as highs``)
and keep ``import gurobipy as gp`` for real Gurobi. Both Model classes are
fully independent.
"""

# ======================================================================
# Pattern A – Drop-in replacement
# ======================================================================

import grbcompat
grbcompat.install()      # patches sys.modules['gurobipy']

import gurobipy as gp            # now resolves to grbcompat
from gurobipy import GRB         # same


def solve_with_highs():
    """Standard gurobipy script – solved by HiGHS via the wrapper."""
    m = gp.Model("knapsack_highs")
    values  = [10, 6, 4, 5, 3]
    weights = [ 5, 4, 3, 2, 1]
    n = len(values)

    x = m.addVars(n, vtype=GRB.BINARY, name="x")
    m.setObjective(gp.quicksum(values[i] * x[i] for i in range(n)), GRB.MAXIMIZE)
    m.addConstr(gp.quicksum(weights[i] * x[i] for i in range(n)) <= 8, "cap")
    m.optimize()

    if m.Status == GRB.OPTIMAL:
        selected = [i for i in range(n) if x[i].X > 0.5]
        print(f"[HiGHS via gurobipy]  obj={m.ObjVal:.0f}  items={selected}")
    else:
        print(f"[HiGHS via gurobipy]  status={m.Status}")


# ======================================================================
# Pattern B – Side-by-side (only when real Gurobi is available)
# ======================================================================

def solve_side_by_side():
    """
    Build the same LP with both HiGHS and real Gurobi and compare results.
    Requires a valid Gurobi licence; silently skipped otherwise.
    """
    import grbcompat as highs
    from grbcompat import GRB as HGRB

    # --- HiGHS model ---
    m_h = highs.Model("lp_highs")
    x_h = m_h.addVar(lb=0.0, name="x")
    y_h = m_h.addVar(lb=0.0, name="y")
    m_h.addConstr(x_h + y_h >= 4, name="demand")
    m_h.setObjective(2 * x_h + 3 * y_h, HGRB.MINIMIZE)
    m_h.optimize()
    print(f"[HiGHS]  x={x_h.X:.4f}  y={y_h.X:.4f}  obj={m_h.ObjVal:.4f}")

    # --- Real Gurobi model (if available) ---
    try:
        # Import the real Gurobi by temporarily bypassing the patch.
        # Because install() already put the wrapper in sys.modules['gurobipy'],
        # we use importlib to load the original package.
        import importlib, sys
        # Remove the patch so importlib finds the real package.
        sys.modules.pop("gurobipy", None)
        real_gp = importlib.import_module("gurobipy")
        GRB_G = real_gp.GRB

        m_g = real_gp.Model("lp_gurobi")
        m_g.setParam("OutputFlag", 0)
        x_g = m_g.addVar(lb=0.0, name="x")
        y_g = m_g.addVar(lb=0.0, name="y")
        m_g.addConstr(x_g + y_g >= 4, name="demand")
        m_g.setObjective(2 * x_g + 3 * y_g, GRB_G.MINIMIZE)
        m_g.optimize()
        print(f"[Gurobi] x={x_g.X:.4f}  y={y_g.X:.4f}  obj={m_g.ObjVal:.4f}")

        delta = abs(m_h.ObjVal - m_g.ObjVal)
        print(f"Objective difference: {delta:.2e}")
    except Exception as exc:
        print(f"[Gurobi] not available ({exc.__class__.__name__})")


if __name__ == "__main__":
    print("=== Pattern A: drop-in replacement ===")
    solve_with_highs()
    print()
    print("=== Pattern B: side-by-side ===")
    solve_side_by_side()
