"""
0-1 Knapsack — Expedition Packing
==================================
You're preparing for a mountain expedition with a 12 kg pack limit.
Each item has a *utility score* and a weight.  Maximize total utility
while staying under the weight budget.

Concepts: binary variables, addVars, quicksum, single constraint MIP.
"""

import grbcompat as gp
from grbcompat import GRB

# (utility score, weight kg)
ITEMS = {
    "tent":          (7, 4.5),
    "sleeping_bag":  (9, 2.1),
    "stove":         (6, 1.2),
    "food_3day":     (8, 3.0),
    "water_3L":      (9, 3.0),
    "first_aid":     (7, 0.5),
    "headlamp":      (5, 0.3),
    "rope":          (4, 2.0),
    "crampons":      (6, 1.8),
    "camera":        (3, 1.5),
    "extra_clothes": (5, 1.0),
    "satellite":     (8, 0.8),
}
CAPACITY = 12.0  # kg


def main():
    names   = list(ITEMS)
    values  = [ITEMS[n][0] for n in names]
    weights = [ITEMS[n][1] for n in names]
    n = len(names)

    m = gp.Model("knapsack")
    m.Params.OutputFlag = 0

    x = m.addVars(n, vtype=GRB.BINARY, name="pack")

    m.setObjective(
        gp.quicksum(values[i] * x[i] for i in range(n)),
        GRB.MAXIMIZE,
    )
    m.addConstr(
        gp.quicksum(weights[i] * x[i] for i in range(n)) <= CAPACITY,
        "capacity",
    )
    m.optimize()

    print("=" * 44)
    print("  Expedition Knapsack  (capacity 12 kg)")
    print("=" * 44)

    if m.Status != GRB.OPTIMAL:
        print(f"Solver status: {m.Status}")
        return

    packed = [(names[i], values[i], weights[i]) for i in range(n) if x[i].X > 0.5]
    left   = [(names[i], values[i], weights[i]) for i in range(n) if x[i].X < 0.5]

    print(f"\n  {'Item':<18} {'Score':>5}  {'kg':>5}")
    print(f"  {'-'*18}  {'-'*5}  {'-'*5}")
    for name, val, wgt in packed:
        print(f"  {name:<18} {val:>5}  {wgt:>5.1f}")
    print(f"  {'─'*18}  {'─'*5}  {'─'*5}")
    total_v = sum(v for _, v, _ in packed)
    total_w = sum(w for _, _, w in packed)
    print(f"  {'PACKED':<18} {total_v:>5}  {total_w:>5.1f}")

    print(f"\n  Left behind: {', '.join(n for n, *_ in left)}")
    print(f"\n  Optimal utility score: {m.ObjVal:.0f}")


if __name__ == "__main__":
    main()
